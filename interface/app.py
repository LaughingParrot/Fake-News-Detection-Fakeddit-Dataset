"""
Multimodal Fake News Detector — Interface
==========================================
"""
from __future__ import annotations

import os
import sys
import json
import time
import platform
import requests
import shutil
import logging
import threading
import warnings
import concurrent.futures
import queue
from datetime import datetime
from io import BytesIO
from typing import Any
from urllib.parse import unquote

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use("TkAgg")          # must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

# ── Optional drag-and-drop (tkinterdnd2) ──────────────────────────────────────
# Not available on all platforms / pip environments; degrade gracefully.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES, DND_TEXT
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False
    DND_FILES = DND_TEXT = None  # type: ignore[assignment]

    class TkinterDnD:  # type: ignore[no-redef]
        """Stub so the rest of the module compiles on platforms without tkinterdnd2."""
        @staticmethod
        def _require(_: Any) -> None:
            pass

# ── Path configuration (anchored to the interface/ folder) ────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_DIR      = os.path.join(APP_DIR, "logs")
APP_LOG_FILE = os.path.join(LOG_DIR, "debug_app.log")
TEMP_IMAGE_DIR = os.path.join(APP_DIR, "_temp_images")
HISTORY_FILE = os.path.join(APP_DIR, "history.json")
MODEL_CKPT_CANDIDATES = (
    os.path.join(APP_DIR,      "multimodal_model.pt"),
    os.path.join(PROJECT_ROOT, "multimodal_model.pt"),
)

os.makedirs(LOG_DIR, exist_ok=True)

# ── Suppress noisy third-party warnings ───────────────────────────────────────
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*QuickGELU.*")
warnings.filterwarnings("ignore", message=".*quick_gelu.*")
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=APP_LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logging.info("=== Application Started ===")

# ── UI theme ──────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Shared colour palette
_C_ROOT   = "#0d1117"
_C_PANEL  = "#161b22"
_C_PANEL2 = "#0f1923"
_C_BORDER = "#1e293b"
_C_ACCENT = "#4f46e5"
_C_BLUE   = "#93c5fd"
_C_TEXT   = "#e2e8f0"
_C_MUTED  = "#64748b"

# Typography — populated in FakeNewsApp.__init__() after a Tk root exists.
# CTkFont requires an active root; creating fonts at module level crashes.
_F_TITLE = _F_HEADING = _F_SUBHEAD = _F_BODY = _F_BUTTON = None
_F_METRIC = _F_MONO = _F_LIVE = _F_STATUS = _F_CAPTION = None
_F_RESULT = _F_OVERLAY = _F_TIMER = _F_PROGRESS = None

# Serialises matplotlib rendering between the main (Tkinter) thread and any
# background export thread so font-manager state is never read/written
# concurrently — prevents the deadlock that caused PDF export to hang.
_MATPLOTLIB_LOCK = threading.Lock()


class FakeNewsApp(ctk.CTk):
    """Main application window for the Multimodal Fake News Detector."""

    def __init__(self) -> None:
        super().__init__(fg_color=_C_ROOT)

        # Enable cross-platform drag-and-drop when tkinterdnd2 is available.
        if _DND_AVAILABLE:
            TkinterDnD._require(self)

        # ── Typography (must be after root creation) ──────────────────────────
        global _F_TITLE, _F_HEADING, _F_SUBHEAD, _F_BODY, _F_BUTTON
        global _F_METRIC, _F_MONO, _F_LIVE, _F_STATUS, _F_CAPTION
        global _F_RESULT, _F_OVERLAY, _F_TIMER, _F_PROGRESS

        _os = platform.system()
        # Primary sans-serif; secondary ensures fallback on unconfigured Linux
        _sans = (
            "Segoe UI"       if _os == "Windows" else
            "SF Pro Display" if _os == "Darwin"  else
            "Ubuntu"
        )
        # Primary monospace; DejaVu Sans Mono ships with most Linux distros
        _mono = (
            "Consolas"       if _os == "Windows" else
            "Menlo"          if _os == "Darwin"  else
            "DejaVu Sans Mono"
        )

        _F_TITLE    = ctk.CTkFont(family=_sans, size=23, weight="bold")
        _F_HEADING  = ctk.CTkFont(family=_sans, size=15, weight="bold")
        _F_SUBHEAD  = ctk.CTkFont(family=_sans, size=13, weight="bold")
        _F_BODY     = ctk.CTkFont(family=_sans, size=12)
        _F_BUTTON   = ctk.CTkFont(family=_sans, size=13, weight="bold")
        _F_METRIC   = ctk.CTkFont(family=_mono, size=13, weight="bold")
        _F_MONO     = ctk.CTkFont(family=_mono, size=13)
        _F_LIVE     = ctk.CTkFont(family=_sans, size=15, slant="italic")
        _F_STATUS   = ctk.CTkFont(family=_sans, size=11)
        _F_CAPTION  = ctk.CTkFont(family=_sans, size=10)
        _F_RESULT   = ctk.CTkFont(family=_sans, size=28, weight="bold")
        _F_OVERLAY  = ctk.CTkFont(family=_sans, size=24, weight="bold")
        _F_TIMER    = ctk.CTkFont(family=_mono, size=22, weight="bold")
        _F_PROGRESS = ctk.CTkFont(family=_sans, size=15, weight="bold")

        self.is_destroyed: bool = False
        self.title("Multimodal Fake News Detector")
        self.configure(bg=_C_ROOT)

        # ── Window geometry ───────────────────────────────────────────────────
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        ww = min(1150, sw - 160)
        wh = min(900,  sh - 160)
        x  = max(0, (sw - ww) // 2)
        y  = max(0, (sh - wh) // 2 - 30)
        self.geometry(f"{ww}x{wh}+{x}+{y}")
        self.minsize(min(1000, ww), min(700, wh))
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── Application state ─────────────────────────────────────────────────
        self.image_path: str | None = None
        self.json_path:  str | None = None
        self.model       = None
        self.tokenizer   = None
        self.image_preprocess = None
        self.device: str = "cpu"

        self.is_processing:    bool = False
        self.cancel_requested: bool = False
        self.start_time:       float = 0.0
        self.batch_run_id:     int   = 0

        self.history_file  = HISTORY_FILE
        self.log_file      = APP_LOG_FILE
        self.json_base_dir = APP_DIR

        self.single_preview_ctk  = None
        self.overlay_preview_ctk = None

        # Prevents Tkinter from garbage-collecting CTkImage objects
        self._image_cache: list = []
        self._after_ids:   list = []

        # Confusion-matrix accumulators (written in main thread via safe_after)
        self.final_tp = self.final_tn = self.final_fp = self.final_fn = 0
        self.final_expert: int = 0          # FIX: was never stored → PNG KPI was always 0

        self.batch_results_data: list[dict] = []
        self.report_text: str = "No analysis report available yet."

        # Protects batch_results_data written from the consumer thread
        self._results_lock = threading.Lock()
        # Thread-local storage for per-worker HTTP sessions
        self._thread_local = threading.local()

        # Producer → Consumer queue
        self.inference_queue: queue.PriorityQueue = queue.PriorityQueue()

        self._initialize_temp_directory()
        self._build_ui()
        self.safe_after(100, self._start_model_load_thread)

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def safe_after(self, ms: int, func: Any, *args: Any) -> None:
        """Schedule *func* on the Tk event loop; silently ignores post-destroy calls."""
        if getattr(self, "is_destroyed", False):
            return

        def _wrapper() -> None:
            if getattr(self, "is_destroyed", False):
                return
            try:
                func(*args)
            except Exception as exc:
                s = str(exc)
                if "invalid command name" not in s and "application has been destroyed" not in s:
                    logging.error(f"Background task error: {exc}")

        try:
            after_id = self.after(ms, _wrapper)
            self._after_ids.append(after_id)
        except Exception:
            pass

    def _initialize_temp_directory(self) -> None:
        """Create (or recreate) the temporary image directory."""
        self.temp_dir = TEMP_IMAGE_DIR
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as exc:
                logging.error(f"Failed to delete old temp dir: {exc}")
        os.makedirs(self.temp_dir, exist_ok=True)
        logging.info(f"Created temp dir: {self.temp_dir}")

    def on_closing(self) -> None:
        """Graceful shutdown — cancel pending work, drain queues, destroy window."""
        logging.info("Application closing.")
        self.is_destroyed    = True
        self.is_processing   = False
        self.cancel_requested = True

        for aid in getattr(self, "_after_ids", []):
            try:
                self.after_cancel(aid)
            except Exception:
                pass

        while not self.inference_queue.empty():
            try:
                self.inference_queue.get_nowait()
                self.inference_queue.task_done()
            except queue.Empty:
                break

        try:
            with _MATPLOTLIB_LOCK:
                plt.close("all")
        except Exception:
            pass

        self._image_cache.clear()
        self.single_preview_ctk  = None
        self.overlay_preview_ctk = None
        self.destroy()
        self.quit()

    # ══════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=7)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="Multimodal Fake News Detector",
            font=_F_TITLE,
            text_color=_C_BLUE,
        ).grid(row=0, column=0, columnspan=2, pady=(14, 8))

        # ── Left panel ────────────────────────────────────────────────────────
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=(16, 8), pady=10, sticky="nsew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure(2, weight=1)

        self._build_single_analysis_block()
        self._build_batch_analysis_block()

        self.bottom_buttons_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.bottom_buttons_frame.grid(row=3, column=0, sticky="w", pady=(10, 5))

        self.history_btn = ctk.CTkButton(
            self.bottom_buttons_frame, text="View History",
            fg_color="#334155", hover_color="#1e293b",
            font=_F_BUTTON, border_width=0, command=self.show_history_window,
        )
        self.history_btn.pack(side="left", padx=(0, 5))

        self.view_logs_btn = ctk.CTkButton(
            self.bottom_buttons_frame, text="View Logs",
            fg_color="#334155", hover_color="#1e293b",
            font=_F_BUTTON, border_width=0, command=self.view_logs,
        )
        self.view_logs_btn.pack(side="left")

        self.status_label = ctk.CTkLabel(
            self.input_frame,
            text="Loading models... Please wait.",
            font=_F_STATUS, text_color="gray",
        )
        self.status_label.grid(row=4, column=0, sticky="w")

        self._build_left_panel_overlay()

        # ── Right panel ───────────────────────────────────────────────────────
        self.output_frame = ctk.CTkFrame(
            self, fg_color=_C_PANEL2, corner_radius=12,
            border_width=1, border_color=_C_BORDER,
        )
        self.output_frame.grid(row=1, column=1, padx=(8, 16), pady=10, sticky="nsew")
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_rowconfigure(1, weight=1)

        self.right_header = ctk.CTkLabel(
            self.output_frame,
            text="Analysis Dashboard",
            font=_F_SUBHEAD, text_color=_C_BLUE,
        )
        self.right_header.grid(row=0, column=0, pady=(14, 8))

        self.view_container = ctk.CTkFrame(self.output_frame, fg_color="transparent")
        self.view_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self.view_container.grid_columnconfigure(0, weight=1)
        self.view_container.grid_rowconfigure(0, weight=1)

        self._build_single_view()
        self._build_batch_view()

        # Show history chart at startup
        self._update_line_chart()
        self.batch_view.grid(row=0, column=0, sticky="nsew")
        self.right_header.configure(text="Analysis Dashboard \u2014 History")

    def _build_single_analysis_block(self) -> None:
        single_frame = ctk.CTkFrame(
            self.input_frame, fg_color=_C_PANEL,
            corner_radius=10, border_width=1, border_color=_C_BORDER,
        )
        single_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10), ipadx=12, ipady=8)

        ctk.CTkLabel(single_frame, text="Single Analysis",
                     font=_F_HEADING, text_color=_C_BLUE).pack(anchor="w", pady=(0, 8))
        ctk.CTkLabel(single_frame, text="News Text",
                     font=_F_BODY, text_color=_C_TEXT).pack(anchor="w")

        self.textbox = ctk.CTkTextbox(single_frame, height=80, corner_radius=6, font=_F_BODY)
        self.textbox.pack(fill="x", pady=(0, 10))
        self.textbox.bind("<KeyRelease>", self._on_textbox_change)

        ctk.CTkLabel(single_frame, text="Associated Image (Optional)",
                     font=_F_BODY, text_color=_C_MUTED).pack(anchor="w")

        self.drop_zone = ctk.CTkFrame(
            single_frame, fg_color="#0d1f33", corner_radius=8,
            cursor="hand2", border_width=1, border_color="#1e3a5f",
        )
        self.drop_zone.pack(fill="x", pady=(0, 5))
        self.drop_zone_label = ctk.CTkLabel(
            self.drop_zone,
            text="Drag & Drop Image Here\nor click to Browse",
            font=_F_CAPTION, text_color="#475569", justify="center",
        )
        self.drop_zone_label.pack(pady=12)

        self.drop_zone.bind("<Button-1>",       lambda _e: self.upload_image())
        self.drop_zone_label.bind("<Button-1>", lambda _e: self.upload_image())
        self.drop_zone.bind("<Enter>", lambda _e: self.drop_zone.configure(
            fg_color="#0d2a4a", border_color="#2e5f8a"))
        self.drop_zone.bind("<Leave>", lambda _e: self.drop_zone.configure(
            fg_color="#0d1f33", border_color="#1e3a5f"))

        # Cross-platform DnD — only register when tkinterdnd2 is installed
        if _DND_AVAILABLE:
            for widget in (self.drop_zone, self.drop_zone_label):
                widget.drop_target_register(DND_FILES, DND_TEXT)
                widget.dnd_bind("<<Drop>>",      self._handle_dnd_event)
                widget.dnd_bind("<<DragEnter>>", lambda _e: self.drop_zone.configure(
                    fg_color="#0d2a4a", border_color="#2e5f8a"))
                widget.dnd_bind("<<DragLeave>>", lambda _e: self.drop_zone.configure(
                    fg_color="#0d1f33", border_color="#1e3a5f"))

        self.img_container = ctk.CTkFrame(single_frame, fg_color="transparent")
        self.img_container.pack(anchor="w", fill="x", pady=(0, 10))
        self.img_label = ctk.CTkLabel(
            self.img_container, text="No image selected",
            font=_F_STATUS, text_color="gray",
        )
        self.img_label.pack(side="left")
        self.remove_img_btn = ctk.CTkButton(
            self.img_container, text="X", width=30,
            fg_color="transparent", hover_color="#7f1d1d",
            command=self.remove_image,
        )

        self.analyze_btn = ctk.CTkButton(
            single_frame, text="Analyze Single News",
            height=38, font=_F_BUTTON,
            fg_color="#4f46e5", hover_color="#4338ca",
            command=self.run_single_analysis, state="disabled",
        )
        self.analyze_btn.pack(fill="x", pady=(5, 0))

    def _build_batch_analysis_block(self) -> None:
        batch_frame = ctk.CTkFrame(
            self.input_frame, fg_color=_C_PANEL,
            corner_radius=10, border_width=1, border_color=_C_BORDER,
        )
        batch_frame.grid(row=1, column=0, sticky="ew", ipadx=12, ipady=8)

        ctk.CTkLabel(batch_frame, text="Batch JSON Analysis",
                     font=_F_HEADING, text_color=_C_BLUE).pack(anchor="w", pady=(0, 8))

        self.upload_json_btn = ctk.CTkButton(
            batch_frame, text="Upload JSON File",
            fg_color="#334155", hover_color="#1e293b", font=_F_BUTTON,
            command=self.upload_json,
        )
        self.upload_json_btn.pack(anchor="w", pady=(0, 5))

        self.json_container = ctk.CTkFrame(batch_frame, fg_color="transparent")
        self.json_container.pack(anchor="w", fill="x", pady=(0, 10))
        self.json_label = ctk.CTkLabel(
            self.json_container, text="No JSON selected",
            font=_F_STATUS, text_color="gray",
        )
        self.json_label.pack(side="left")
        self.remove_json_btn = ctk.CTkButton(
            self.json_container, text="X", width=30,
            fg_color="transparent", hover_color="#7f1d1d",
            command=self.remove_json,
        )

        self.batch_analyze_btn = ctk.CTkButton(
            batch_frame, text="Run Batch Analysis",
            height=38, font=_F_BUTTON,
            fg_color="#4f46e5", hover_color="#4338ca",
            command=self.run_batch_analysis, state="disabled",
        )
        self.batch_analyze_btn.pack(fill="x", pady=(5, 0))

    def _build_left_panel_overlay(self) -> None:
        self.overlay = ctk.CTkFrame(
            self.input_frame, fg_color=("gray85", "gray20"), corner_radius=10,
        )
        self.overlay.bind("<Button-1>", lambda _e: "break")

        self.overlay_loader = ctk.CTkLabel(
            self.overlay, text="Processing...", font=_F_OVERLAY,
        )
        self.overlay_loader.place(relx=0.5, rely=0.30, anchor="center")

        self.overlay_timer = ctk.CTkLabel(
            self.overlay, text="00:00", font=_F_TIMER, text_color="#3498db",
        )
        self.overlay_timer.place(relx=0.5, rely=0.40, anchor="center")

        self.overlay_img_label = ctk.CTkLabel(self.overlay, text="", corner_radius=10)
        self.overlay_img_label.place(relx=0.5, rely=0.70, anchor="center")

    def _build_single_view(self) -> None:
        self.single_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        self.result_label = ctk.CTkLabel(self.single_view, text="-", font=_F_RESULT)
        self.result_label.pack(pady=40)
        self.conf_label = ctk.CTkLabel(
            self.single_view, text="Confidence: -", font=_F_METRIC,
        )
        self.conf_label.pack(pady=5)
        self.warning_label = ctk.CTkLabel(
            self.single_view, text="", text_color="orange",
            font=_F_STATUS, wraplength=400,
        )
        self.warning_label.pack(pady=20)

    def _build_batch_view(self) -> None:
        self.batch_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        self.batch_view.grid_columnconfigure(0, weight=1)
        for row, weight in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 0), (6, 0)]:
            self.batch_view.grid_rowconfigure(row, weight=weight)

        self.realtime_progress_label = ctk.CTkLabel(
            self.batch_view, text="", font=_F_PROGRESS,
        )
        self.realtime_progress_label.grid(row=0, column=0, pady=(0, 3))

        self.live_headline_box = ctk.CTkTextbox(
            self.batch_view, height=55, font=_F_LIVE,
            text_color="#3498db", state="disabled",
        )
        self.live_action_label = ctk.CTkLabel(
            self.batch_view, text="", font=_F_STATUS, text_color="#f39c12",
        )

        self.metrics_block = ctk.CTkFrame(
            self.batch_view, fg_color="#0d2137",
            corner_radius=8, border_width=1, border_color=_C_BORDER,
        )
        self.realtime_metrics_label = ctk.CTkLabel(
            self.metrics_block,
            text="Acc: \u2014  |  Prec: \u2014  |  Rec: \u2014  |  F1: \u2014",
            font=_F_METRIC, text_color="#93c5fd",
        )
        self.realtime_metrics_label.pack(pady=(7, 2))
        self.expert_count_label = ctk.CTkLabel(
            self.metrics_block,
            text="Expert Review Triggered: 0 items",
            font=_F_METRIC, text_color="#93c5fd",
        )
        self.expert_count_label.pack(pady=(0, 7))

        # Matplotlib — 2-panel: donut (left) + history line chart (right)
        # constrained_layout avoids tight_layout/aspect='equal' warnings
        with _MATPLOTLIB_LOCK:
            self.fig, (self.ax_pie, self.ax_line) = plt.subplots(
                1, 2, figsize=(9, 4.2), dpi=90,
                gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.35},
                constrained_layout=True,
            )
            self.fig.patch.set_facecolor(_C_ROOT)
            self.ax_pie.set_facecolor(_C_ROOT)
            self.ax_line.set_facecolor(_C_ROOT)

        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.batch_view)
        self.graph_canvas.get_tk_widget().grid(row=4, column=0, sticky="nsew", pady=(5, 10))

        self.pie_labels = ["TP", "TN", "FP", "FN"]
        self.pie_colors = ["#22c55e", "#3b82f6", "#ef4444", "#f59e0b"]

        # ── Action buttons ────────────────────────────────────────────────────
        self.action_frame = ctk.CTkFrame(self.batch_view, fg_color="transparent")
        self.action_frame.grid_columnconfigure(0, weight=1)

        self.export_btn = ctk.CTkButton(
            self.action_frame,
            text="Export Report & Graphs",
            height=40, font=_F_BUTTON,
            fg_color="#4f46e5", hover_color="#4338ca",
            command=self.export_results,
        )
        self.export_btn.grid(row=0, column=0, sticky="ew")

        # FIX: was action_frame.lower() — lower() only changes z-order, does NOT hide.
        # Use grid_remove() to actually hide until batch completes.
        self.action_frame.grid_remove()

        self.stop_btn = ctk.CTkButton(
            self.batch_view, text="Stop Analysis",
            fg_color="#7f1d1d", hover_color="#991b1b",
            height=35, font=_F_BUTTON,
            command=self.stop_batch_analysis,
        )
        self.stop_btn.grid(row=6, column=0, sticky="ew", pady=(0, 5))
        self.stop_btn.grid_remove()

    # ══════════════════════════════════════════════════════════════════════════
    # BACKGROUND MODEL LOADING
    # ══════════════════════════════════════════════════════════════════════════

    def _start_model_load_thread(self) -> None:
        if self.model is not None:
            return
        threading.Thread(target=self._load_model_heavy, daemon=True).start()

    def _load_model_heavy(self) -> None:
        try:
            import torch
            import open_clip
            from models.multimodal_model import FakeNewsMultimodalModel

            logging.info("Loading PyTorch and OpenCLIP models...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name  = "ViT-B-32"

            self.tokenizer = open_clip.get_tokenizer(model_name)
            _, _, self.image_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai",
            )

            self.model = FakeNewsMultimodalModel(
                freeze_text_encoder=True,
                freeze_image_encoder=True,
            )

            checkpoint = next(
                (p for p in MODEL_CKPT_CANDIDATES if os.path.exists(p)),
                MODEL_CKPT_CANDIDATES[-1],
            )
            try:
                self.model.load_state_dict(
                    torch.load(checkpoint, map_location=self.device, weights_only=True)
                )
                status = f"Models Ready (Device: {self.device.upper()})"
                logging.info(f"Loaded checkpoint from {checkpoint} on {self.device}")
            except FileNotFoundError:
                status = "Models Ready (Warning: untrained weights)"
                logging.warning(f"multimodal_model.pt not found: {MODEL_CKPT_CANDIDATES}")

            self.model.to(self.device)
            self.model.eval()
            self.safe_after(0, self._enable_ui, status)

        except Exception as exc:
            logging.error(f"Error loading model: {exc}")
            self.safe_after(0, lambda: self.status_label.configure(
                text="Error loading models", text_color="#ef4444",
            ))

    def _enable_ui(self, status_text: str) -> None:
        colour = "#22c55e" if ("Ready" in status_text and "Warning" not in status_text) else "orange"
        self.status_label.configure(text=status_text, text_color=colour)
        self.analyze_btn.configure(state="normal")
        self.batch_analyze_btn.configure(state="normal")

    # ══════════════════════════════════════════════════════════════════════════
    # INPUT HANDLING
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_dnd_event(self, event: Any) -> None:
        """Universal drop handler for tkinterdnd2 (files, file:// URIs, http URLs, HTML)."""
        import re
        data   = event.data.strip()
        tokens = [t.strip() for t in data.replace("\r", "\n").split("\n") if t.strip()]

        for token in tokens:
            if token.startswith("#"):
                continue
            if token.lower().startswith(("http://", "https://")):
                self._load_image_from_url(token)
                return
            if token.lower().startswith("file://"):
                path = token[7:]
                if os.name == "nt" and path.startswith("/"):
                    path = path[1:]
                self._handle_drop(unquote(path))
                return
            path = token.strip("{}")
            if os.path.isfile(path):
                self._handle_drop(path)
                return

        # HTML fragment: <img src="…">
        match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', data, re.IGNORECASE)
        if match:
            src = match.group(1)
            if src.lower().startswith(("http://", "https://")):
                self._load_image_from_url(src)
            elif src.lower().startswith("file://"):
                path = unquote(src[7:])
                if os.name == "nt" and path.startswith("/"):
                    path = path[1:]
                self._handle_drop(path)

        logging.info(f"Unrecognised drop payload: {data[:120]}")

    def _handle_drop(self, path: str) -> None:
        path = path.strip(' \n\r\t"\'{}')
        ext  = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
            self._load_single_image(path)
        elif ext == ".json":
            logging.info(f"JSON dropped: {path}")
            self.json_path     = path
            self.json_base_dir = os.path.dirname(os.path.abspath(path))
            self.json_label.configure(text=os.path.basename(path), text_color="white")
            self.remove_json_btn.pack(side="left", padx=(10, 0))
            self.status_label.configure(text="", text_color="gray")
        else:
            logging.info(f"Dropped file ignored (not image/json): {path}")

    def _load_image_from_url(self, url: str) -> None:
        """Download a URL image on a daemon thread; loads result into drop zone."""
        if hasattr(self, "drop_zone_label"):
            self.drop_zone_label.configure(text="Downloading image\u2026", text_color="#f59e0b")

        def _download() -> None:
            try:
                sess = requests.Session()
                sess.headers["User-Agent"] = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
                sess.verify = False
                resp = sess.get(url, timeout=12)
                resp.raise_for_status()
                img  = Image.open(BytesIO(resp.content)).convert("RGB")
                fname = os.path.basename(url.split("?")[0]) or "dropped_url"
                if not os.path.splitext(fname)[1]:
                    fname += ".jpg"
                tmp = os.path.join(self.temp_dir, fname)
                img.save(tmp, "JPEG", quality=85, optimize=True)
                logging.info(f"URL image saved to: {tmp}")
                self.safe_after(0, self._load_single_image, tmp)
            except Exception as exc:
                logging.warning(f"URL download failed '{url}': {exc}")
                self.safe_after(0, self._on_url_download_failed)

        threading.Thread(target=_download, daemon=True).start()

    def _on_url_download_failed(self) -> None:
        if hasattr(self, "drop_zone_label"):
            self.drop_zone_label.configure(
                text="Could not download image from URL", text_color="#ef4444",
            )

    def upload_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")]
        )
        if path:
            self._load_single_image(path)

    def _load_single_image(self, file_path: str) -> None:
        """Load any image into the preview; save 224×224 JPEG to temp dir for inference."""
        logging.info(f"Single image loaded: {file_path}")
        try:
            img = Image.open(file_path).convert("RGB")

            model_img = img.resize((224, 224), Image.Resampling.LANCZOS)
            comp_path = os.path.join(self.temp_dir, "single_upload.jpg")
            model_img.save(comp_path, "JPEG", quality=85, optimize=True)
            self.image_path = comp_path

            preview = img.copy()
            preview.thumbnail((120, 120), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(
                light_image=preview, dark_image=preview,
                size=(preview.width, preview.height),
            )
            self._image_cache.append(ctk_img)
            if len(self._image_cache) > 10:
                self._image_cache.pop(0)

            self.single_preview_ctk = ctk_img
            self.img_label.configure(text="", image=ctk_img)
            self.img_label.update()
            self.remove_img_btn.pack(side="left", padx=(10, 0))
            if hasattr(self, "drop_zone_label"):
                self.drop_zone_label.configure(
                    text=os.path.basename(file_path), text_color="#22c55e",
                )
            self.status_label.configure(text="", text_color="gray")

        except (UnidentifiedImageError, OSError) as exc:
            logging.warning(f"Invalid single image: {exc}")
            self.image_path = None
            empty = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))
            self.img_label.configure(
                text="Invalid image selected", image=empty, text_color="#ef4444",
            )
            self.img_label.update()
            self.single_preview_ctk = None
            self.remove_img_btn.pack_forget()
            if hasattr(self, "drop_zone_label"):
                self.drop_zone_label.configure(
                    text="Invalid image file", text_color="#ef4444",
                )

    def remove_image(self) -> None:
        self.image_path         = None
        self.single_preview_ctk = None
        empty = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))
        self.img_label.configure(text="No image selected", image=empty, text_color="gray")
        self.img_label.update()
        self.remove_img_btn.pack_forget()
        if hasattr(self, "drop_zone_label"):
            self.drop_zone_label.configure(
                text="Drag & Drop Image Here\nor click to Browse",
                text_color="#6b7280",
            )
        self.status_label.configure(text="", text_color="gray")
        logging.info("Single image removed by user.")

    def upload_json(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            logging.info(f"JSON uploaded: {path}")
            self.json_path     = path
            self.json_base_dir = os.path.dirname(os.path.abspath(path))
            self.json_label.configure(text=os.path.basename(path), text_color="white")
            self.remove_json_btn.pack(side="left", padx=(10, 0))
            self.status_label.configure(text="", text_color="gray")

    def remove_json(self) -> None:
        self.json_path     = None
        self.json_base_dir = APP_DIR
        self.json_label.configure(text="No JSON selected", text_color="gray")
        self.remove_json_btn.pack_forget()
        self.status_label.configure(text="", text_color="gray")
        logging.info("JSON file removed by user.")

    # ══════════════════════════════════════════════════════════════════════════
    # OVERLAY
    # ══════════════════════════════════════════════════════════════════════════

    def _show_overlay(self) -> None:
        self.is_processing = True
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.start_time = time.time()
        self.overlay_preview_ctk = None
        empty = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))
        self.overlay_img_label.configure(image=empty, text="")
        self._update_timer()

    def _hide_overlay(self) -> None:
        self.is_processing = False
        self.overlay.place_forget()

    def _update_timer(self) -> None:
        if not self.is_processing:
            return
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        self.overlay_timer.configure(text=f"{m:02d}:{s:02d}")
        self.safe_after(1000, self._update_timer)

    def _update_translucent_image(self, pil_img: Image.Image | None) -> None:
        if pil_img:
            img = pil_img.copy()
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            img = img.resize((240, 240), Image.Resampling.LANCZOS)
            alpha = img.split()[3].point(lambda p: int(p * 0.35))
            img.putalpha(alpha)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(240, 240))
            self._image_cache.append(ctk_img)
            if len(self._image_cache) > 10:
                self._image_cache.pop(0)
            self.overlay_preview_ctk = ctk_img
            self.overlay_img_label.configure(image=ctk_img, text="")
        else:
            self.overlay_preview_ctk = None
            empty = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))
            self.overlay_img_label.configure(
                image=empty, text="Image Unavailable", text_color="#ef4444",
            )

    def _update_live_headline(self, text: str) -> None:
        self.live_headline_box.configure(state="normal")
        self.live_headline_box.delete("1.0", "end")
        self.live_headline_box.insert("1.0", f'"{text}"')
        self.live_headline_box.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    # CORE ML INFERENCE
    # ══════════════════════════════════════════════════════════════════════════

    def _get_probabilities(self, text: str, img_path: str | None) -> tuple[float, float]:
        """Run text + image through model; returns (prob_real, prob_fake)."""
        import torch

        tokens = self.tokenizer([text])[0].unsqueeze(0).to(self.device)

        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as pil:
                image_proc = pil.convert("RGB")
        else:
            image_proc = Image.new("RGB", (224, 224), (0, 0, 0))

        image_tensor = self.image_preprocess(image_proc).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _, _, _ = self.model(tokens, image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return float(probs[0].item()), float(probs[1].item())

    # ══════════════════════════════════════════════════════════════════════════
    # SINGLE ANALYSIS PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def run_single_analysis(self) -> None:
        text_input = self.textbox.get("1.0", "end").strip()
        if not text_input:
            self.status_label.configure(
                text="Please enter news text first.", text_color="orange",
            )
            return

        logging.info("Starting single analysis.")
        # FIX: disable buttons during processing to prevent double-click race
        self.analyze_btn.configure(state="disabled")
        self.batch_analyze_btn.configure(state="disabled")

        self.batch_view.grid_forget()
        self.single_view.grid(row=0, column=0, sticky="nsew")
        self.right_header.configure(text="Analysis Processing...")
        self._show_overlay()

        threading.Thread(
            target=self._single_inference_task,
            args=(text_input, self.image_path),
            daemon=True,
        ).start()

    def _single_inference_task(self, text: str, img_path: str | None) -> None:
        try:
            prob_real, prob_fake = self._get_probabilities(text, img_path)
            logging.info(f"Single analysis: Real={prob_real:.2f}, Fake={prob_fake:.2f}")
            self.safe_after(0, self._update_single_results, prob_real, prob_fake)
        except Exception as exc:
            logging.error(f"Single inference error: {exc}")
            self.safe_after(0, lambda: self.right_header.configure(text="Analysis Results"))
            self.safe_after(0, self._hide_overlay)
            self.safe_after(0, lambda: self.analyze_btn.configure(state="normal"))
            self.safe_after(0, lambda: self.batch_analyze_btn.configure(state="normal"))

    def _on_textbox_change(self, _event: Any = None) -> None:
        """Clear the validation warning when the user starts typing."""
        current = self.status_label.cget("text")
        if "please enter news text" in current.lower():
            if self.model is not None:
                dev = getattr(self, "device", "cpu").upper()
                self.status_label.configure(
                    text=f"Models Ready (Device: {dev})", text_color="#22c55e",
                )
            else:
                self.status_label.configure(text="", text_color="gray")

    def _update_single_results(self, prob_real: float, prob_fake: float) -> None:
        self._hide_overlay()
        # Re-enable buttons after processing
        self.analyze_btn.configure(state="normal")
        self.batch_analyze_btn.configure(state="normal")

        self.right_header.configure(text="Analysis Results")
        conf = max(prob_real, prob_fake) * 100

        if prob_fake > prob_real:
            self.result_label.configure(text="FAKE NEWS", text_color="#ef4444")
        else:
            self.result_label.configure(text="REAL NEWS", text_color="#22c55e")

        self.conf_label.configure(text=f"Confidence: {conf:.2f}%")
        self.warning_label.configure(
            text="Low confidence. Expert review recommended." if conf < 65.0 else "",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # BATCH PRODUCER-CONSUMER PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def run_batch_analysis(self) -> None:
        if self.model is None or self.tokenizer is None or self.image_preprocess is None:
            self.status_label.configure(
                text="Model not ready yet. Please wait.", text_color="orange",
            )
            return
        if not self.json_path:
            self.status_label.configure(
                text="Please upload a JSON file.", text_color="orange",
            )
            return

        logging.info(f"Starting batch analysis: {self.json_path}")

        # FIX: disable buttons during processing
        self.analyze_btn.configure(state="disabled")
        self.batch_analyze_btn.configure(state="disabled")

        self.single_view.grid_forget()
        self.batch_view.grid(row=0, column=0, sticky="nsew")
        self.right_header.configure(text="Batch Processing Pipeline...")

        self.cancel_requested = False
        # FIX: action_frame was action_frame.lower() (z-order only, not hidden).
        # grid_remove() actually hides the widget.
        self.action_frame.grid_remove()
        self.metrics_block.grid_remove()

        with _MATPLOTLIB_LOCK:
            self.ax_line.set_visible(False)
            self.ax_pie.set_visible(True)
            self.ax_pie.set_position([0.08, 0.10, 0.84, 0.80])
            self.ax_pie.clear()
            self.ax_pie.set_facecolor(_C_ROOT)
            self.ax_pie.set_aspect("equal")
            self.ax_pie.pie(
                [1], colors=["#161b22"], radius=1,
                wedgeprops=dict(width=0.46, edgecolor=_C_ROOT, linewidth=2),
            )
            self.ax_pie.text(0, 0, "Starting...", ha="center", va="center",
                             color="#334155", fontsize=13, weight="bold")
            self.ax_pie.set_title("Live Prediction Distribution",
                                  color="#93c5fd", pad=12, weight="bold", fontsize=11)
            self.graph_canvas.draw_idle()

        self.stop_btn.configure(state="normal", text="Stop Analysis")
        self.stop_btn.grid()

        self.live_headline_box.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.live_action_label.grid(row=1, column=0, pady=(0, 4))
        self.metrics_block.grid(row=2, column=0, sticky="ew", pady=(0, 6), padx=4)
        self.realtime_metrics_label.configure(
            text="Acc: \u2014  |  Prec: \u2014  |  Rec: \u2014  |  F1: \u2014"
        )
        self.expert_count_label.configure(text="Expert Review Triggered: 0 items")
        self.realtime_progress_label.configure(text="Preparing batch...")

        self.inference_queue = queue.PriorityQueue()
        self.batch_run_id   += 1
        current_run_id = self.batch_run_id
        self._show_overlay()

        threading.Thread(
            target=self._batch_manager_thread,
            args=(current_run_id,),
            daemon=True,
        ).start()

    def stop_batch_analysis(self) -> None:
        self.cancel_requested = True
        self.live_action_label.configure(
            text="Stopping\u2026 finishing current items.", text_color="#ef4444",
        )
        self.stop_btn.configure(state="disabled", text="Stopping...")
        logging.info("User stopped batch analysis.")

    def _get_thread_session(self) -> requests.Session:
        """Return a thread-local requests.Session with connection pooling."""
        if not hasattr(self._thread_local, "session"):
            sess = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8, pool_maxsize=8, max_retries=2,
            )
            sess.mount("http://",  adapter)
            sess.mount("https://", adapter)
            sess.headers["User-Agent"] = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            sess.verify = False
            self._thread_local.session = sess
        return self._thread_local.session

    def _batch_manager_thread(self, run_id: int) -> None:
        """Submit download tasks via thread pool; signal consumer when done."""
        consumer_started = False
        futures: list = []
        executor = None

        try:
            if run_id != self.batch_run_id:
                return

            with open(self.json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            if not isinstance(data, list):
                raise ValueError("Batch JSON root must be a list of items.")

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            for idx, item in enumerate(data):
                if self.cancel_requested or run_id != self.batch_run_id:
                    break
                futures.append(executor.submit(self._download_producer_task, item, idx, run_id))

            submitted = len(futures)
            if submitted == 0:
                self.safe_after(0, lambda: self.realtime_progress_label.configure(
                    text="No items to process.",
                ))
                self.safe_after(0, self._finalize_batch, 0, 0, 0, 0, 0, 0)
                return

            self.safe_after(0, lambda: self.realtime_progress_label.configure(
                text=f"Processing 0/{submitted}",
            ))
            threading.Thread(
                target=self._inference_consumer_thread,
                args=(submitted, run_id),
                daemon=True,
            ).start()
            consumer_started = True

            for future in concurrent.futures.as_completed(futures):
                if self.cancel_requested or run_id != self.batch_run_id:
                    break
                try:
                    future.result()
                except Exception:
                    logging.exception("Producer task crashed unexpectedly.")

        except Exception as exc:
            logging.error(f"Batch manager error: {exc}")
            if run_id == self.batch_run_id:
                self.safe_after(0, self._hide_overlay)
                self.safe_after(0, self.stop_btn.grid_remove)
                self.safe_after(0, lambda: self.analyze_btn.configure(state="normal"))
                self.safe_after(0, lambda: self.batch_analyze_btn.configure(state="normal"))
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if consumer_started:
                self.inference_queue.put(((99, 10 ** 12), {"_done": True, "run_id": run_id}))

    def _resolve_local_image_path(self, image_ref: str | None) -> str | None:
        if not image_ref:
            return None
        candidate = str(image_ref).strip()
        if not candidate:
            return None

        if candidate.lower().startswith("file://"):
            candidate = unquote(candidate[7:])
            if os.name == "nt" and candidate.startswith("/") and len(candidate) > 2 and candidate[2] == ":":
                candidate = candidate[1:]

        candidate = os.path.expanduser(candidate)

        if os.path.isabs(candidate):
            p = os.path.abspath(candidate)
            return p if os.path.isfile(p) else None

        base = getattr(self, "json_base_dir", APP_DIR)
        rel_json = os.path.abspath(os.path.join(base, candidate))
        if os.path.isfile(rel_json):
            return rel_json

        rel_cwd = os.path.abspath(candidate)
        return rel_cwd if os.path.isfile(rel_cwd) else None

    def _save_processed_image(self, image_obj: Image.Image, output_path: str) -> Image.Image:
        processed = image_obj.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        processed.save(output_path, "JPEG", quality=40, optimize=True)
        return processed.copy()

    def _download_producer_task(self, item: Any, idx: int, run_id: int) -> None:
        if not self.is_processing or self.cancel_requested or run_id != self.batch_run_id:
            return

        item_dict  = item if isinstance(item, dict) else {"id": "unknown", "text": "", "label": 0}
        item_id    = str(item_dict.get("id", "unknown"))
        safe_id    = item_id.replace("\\", "_").replace("/", "_").replace(":", "_")

        payload: dict = {
            "item": item_dict, "img_path": None,
            "display_img": None, "error_msg": None, "run_id": run_id,
        }

        try:
            if not isinstance(item, dict):
                payload["error_msg"] = "[Schema Error] Item is not a JSON object."
            else:
                image_ref = next(
                    (v for v in (
                        item_dict.get("image_url"),
                        item_dict.get("image_path"),
                        item_dict.get("image"),
                    ) if isinstance(v, str) and v.strip()),
                    "",
                ).strip()

                if not image_ref:
                    payload["error_msg"] = "[Skipped] No image reference found."
                elif image_ref.lower().startswith(("http://", "https://")):
                    img_path = os.path.join(self.temp_dir, f"{safe_id}.jpg")
                    sess = self._get_thread_session()
                    for attempt in range(4):
                        if self.cancel_requested or run_id != self.batch_run_id:
                            break
                        resp = sess.get(image_ref, timeout=(3.5, 30.0))
                        if resp.status_code in (429, 503):
                            time.sleep(3 * (2 ** attempt))
                            continue
                        resp.raise_for_status()
                        if len(resp.content) < 500:
                            raise ValueError("Payload too small (broken link)")
                        with Image.open(BytesIO(resp.content)) as raw:
                            payload["display_img"] = self._save_processed_image(raw, img_path)
                        payload["img_path"] = img_path
                        break

                    if (payload["img_path"] is None
                            and payload["error_msg"] is None
                            and not self.cancel_requested
                            and run_id == self.batch_run_id):
                        payload["error_msg"] = "[Network/File Error] Image unavailable"
                else:
                    local = self._resolve_local_image_path(image_ref)
                    if local is None:
                        payload["error_msg"] = "[Path Error] Local image not found"
                        logging.warning(f"Local image not found for {item_id}: {image_ref}")
                    else:
                        img_path = os.path.join(self.temp_dir, f"{safe_id}.jpg")
                        with Image.open(local) as raw:
                            payload["display_img"] = self._save_processed_image(raw, img_path)
                        payload["img_path"] = img_path

        except Exception as exc:
            payload.update(img_path=None, display_img=None,
                           error_msg="[Network/File Error] Image unavailable")
            logging.warning(f"Download failed for {item_id}: {exc}")

        if not self.cancel_requested and run_id == self.batch_run_id:
            priority = 0 if payload["img_path"] else 1
            self.inference_queue.put(((priority, idx), payload))

    def _inference_consumer_thread(self, total_items: int, run_id: int) -> None:
        """Sequential inference loop; stops on _done sentinel from manager."""
        tp = tn = fp = fn = expert = 0
        local_results: list[dict] = []
        processed = 0

        while True:
            try:
                _, payload = self.inference_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if payload.get("run_id") != run_id:
                self.inference_queue.task_done()
                continue

            if payload.get("_done"):
                self.inference_queue.task_done()
                break

            processed += 1
            item       = payload.get("item", {})
            text       = str(item.get("text", ""))
            try:
                true_label = int(item.get("label", 0))
            except (TypeError, ValueError):
                true_label = 0

            self.safe_after(0, lambda c=processed, t=total_items:
                self.realtime_progress_label.configure(text=f"Processing {c}/{t}"))
            self.safe_after(0, lambda tx=text: self._update_live_headline(tx))

            if payload.get("error_msg"):
                self.safe_after(0, lambda e=payload["error_msg"]:
                    self.live_action_label.configure(text=e, text_color="#ef4444"))
                self.safe_after(0, lambda: self._update_translucent_image(None))
            else:
                self.safe_after(0, lambda: self.live_action_label.configure(
                    text="[Success] Image Optimized & Loaded", text_color="#22c55e"))
                self.safe_after(0, lambda p=payload.get("display_img"):
                    self._update_translucent_image(p))

            try:
                prob_real, prob_fake = self._get_probabilities(text, payload.get("img_path"))
                pred       = 1 if prob_fake > prob_real else 0
                confidence = max(prob_real, prob_fake)

                if   pred == 1 and true_label == 1: tp += 1
                elif pred == 0 and true_label == 0: tn += 1
                elif pred == 1 and true_label == 0: fp += 1
                elif pred == 0 and true_label == 1: fn += 1

                if confidence < 0.65:
                    expert += 1

                local_results.append({
                    "id":              item.get("id"),
                    "text":            text,
                    "image_url":       (item.get("image_url")
                                        or item.get("image_path")
                                        or item.get("image") or ""),
                    "img_path":        payload.get("img_path"),
                    "predicted":       pred,
                    "predicted_label": "Fake" if pred == 1 else "Real",
                    "true":            true_label,
                    "true_label":      "Fake" if true_label == 1 else "Real",
                    "confidence":      round(confidence, 4),
                    "expert_review":   confidence < 0.65,
                })

                correct     = tp + tn
                total_so_far = correct + fp + fn
                acc  = correct / total_so_far if total_so_far > 0 else 0
                prec = tp / (tp + fp)         if (tp + fp) > 0   else 0
                rec  = tp / (tp + fn)         if (tp + fn) > 0   else 0
                f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

                self.safe_after(0, lambda a=acc, p=prec, r=rec, f=f1, ex=expert: [
                    self.realtime_metrics_label.configure(
                        text=f"Acc: {a:.4f}  |  Prec: {p:.4f}  |  Rec: {r:.4f}  |  F1: {f:.4f}"
                    ),
                    self.expert_count_label.configure(
                        text=f"Expert Review Triggered: {ex} items"
                    ),
                ])
                self.safe_after(0, self._fast_update_pie_chart,
                                tp, tn, fp, fn, processed, total_items)

            except Exception as exc:
                logging.error(f"Inference error for item {item.get('id')}: {exc}")

            self.inference_queue.task_done()

        with self._results_lock:
            self.batch_results_data = local_results

        if self.is_processing and run_id == self.batch_run_id:
            self.safe_after(0, self._finalize_batch, tp, tn, fp, fn, expert, processed)

    def _fast_update_pie_chart(self, tp: int, tn: int, fp: int, fn: int,
                                current: int, total: int) -> None:
        with _MATPLOTLIB_LOCK:
            self.ax_pie.clear()
            self.ax_pie.set_position([0.08, 0.10, 0.84, 0.80])
            self.ax_pie.set_facecolor("#111827")
            self.ax_pie.set_aspect("equal")

            vals   = [tp, tn, fp, fn]
            labels = self.pie_labels
            colors = self.pie_colors

            filtered = [(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0]
            if filtered:
                fv, fl, fc = zip(*filtered)
                wedges, _, autotexts = self.ax_pie.pie(
                    fv, labels=fl, colors=fc, autopct="%1.1f%%",
                    textprops={"color": "white", "fontsize": 9.5, "weight": "bold"},
                    radius=1,
                    wedgeprops=dict(width=0.45, edgecolor="#111827", linewidth=1.5),
                    startangle=90,
                )
                for at in autotexts:
                    at.set_fontsize(8.5)
            else:
                self.ax_pie.pie([1], colors=["#1f2937"], radius=1,
                                wedgeprops=dict(width=0.45, edgecolor="#111827"))
                self.ax_pie.text(0, 0, "Ready", ha="center", va="center",
                                 color="#6b7280", fontsize=14, weight="bold")

            self.ax_pie.text(0, 0, f"{current}\n/{total}", ha="center", va="center",
                             color="white", fontsize=13, weight="bold")
            self.ax_pie.set_title("Live Prediction Distribution",
                                  color="#93c5fd", pad=12, weight="bold", fontsize=11)
            self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
            self.graph_canvas.draw_idle()

    def _finalize_batch(self, tp: int, tn: int, fp: int, fn: int,
                         expert: int, total: int) -> None:
        self._hide_overlay()
        logging.info("Batch processing complete.")

        self.final_tp = tp
        self.final_tn = tn
        self.final_fp = fp
        self.final_fn = fn
        self.final_expert = expert          # FIX: was never stored

        correct      = tp + tn
        accuracy     = correct / total        if total       > 0 else 0.0
        precision    = tp / (tp + fp)         if (tp + fp)  > 0 else 0.0
        recall       = tp / (tp + fn)         if (tp + fn)  > 0 else 0.0
        f1           = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        expert_ratio = expert / total         if total       > 0 else 0.0

        self.right_header.configure(text="Batch Analysis Results")
        if self.cancel_requested:
            self.realtime_progress_label.configure(
                text=f"Processing Stopped ({total} items processed).")
        else:
            self.realtime_progress_label.configure(
                text=f"Processing Complete \u2014 {total} items.")

        self.stop_btn.grid_remove()
        self.live_headline_box.grid_remove()
        self.live_action_label.grid_remove()

        self.realtime_metrics_label.configure(
            text=f"Acc: {accuracy:.4f}  |  Prec: {precision:.4f}  |  Rec: {recall:.4f}  |  F1: {f1:.4f}"
        )
        self.expert_count_label.configure(
            text=f"Expert Review Triggered: {expert} items  ({expert_ratio:.1%})"
        )
        self.metrics_block.grid(row=3, column=0, sticky="ew", pady=(0, 6), padx=4)

        # FIX: action_frame.lift() only changes z-order, doesn't show a grid_remove'd widget.
        # Restore with .grid() using the same options as initial layout.
        self.action_frame.grid(row=5, column=0, sticky="ew", pady=(0, 5))

        # Re-enable buttons now that batch is done
        self.analyze_btn.configure(state="normal")
        self.batch_analyze_btn.configure(state="normal")

        self.ax_line.set_visible(True)

        raw_time = datetime.now()
        record = {
            "time": raw_time.strftime("%I:%M %p").lower().replace("am", "a.m.").replace("pm", "p.m."),
            "date": raw_time.strftime("%d %b %Y"),
            "metrics": {
                "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1,
                "expert_ratio": expert_ratio,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            },
        }
        self.save_to_history(record)
        self._update_line_chart()

        self.report_text = (
            f"Confusion Matrix Summary:\n"
            f"---------------------------------------------\n"
            f"True Positive  (Fake -> Fake) : {tp}\n"
            f"True Negative  (Real -> Real) : {tn}\n"
            f"False Positive (Real -> Fake) : {fp}\n"
            f"False Negative (Fake -> Real) : {fn}\n\n"
            f"Performance Metrics:\n"
            f"---------------------------------------------\n"
            f"Accuracy : {accuracy:.4f}    Precision: {precision:.4f}\n"
            f"Recall   : {recall:.4f}    F1 Score : {f1:.4f}\n\n"
            f"Expert Review Triggered : {expert} items ({expert_ratio:.2%})\n"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY CHART & DATA
    # ══════════════════════════════════════════════════════════════════════════

    def _update_line_chart(self) -> None:
        history = self.load_history()

        BG      = _C_ROOT
        ACC_C   = "#22c55e"
        PREC_C  = "#3b82f6"
        REC_C   = "#f59e0b"
        F1_C    = "#a78bfa"
        GRID_C  = "#1e293b"
        TEXT_C  = "#94a3b8"
        TITLE_C = "#93c5fd"

        no_results = (self.final_tp + self.final_tn + self.final_fp + self.final_fn) == 0

        with _MATPLOTLIB_LOCK:
            # ── Left: donut ──────────────────────────────────────────────────
            self.ax_pie.clear()
            self.ax_pie.set_facecolor(BG)

            if no_results:
                self.ax_pie.set_visible(False)
                self.ax_line.set_position([0.07, 0.12, 0.90, 0.76])
            else:
                self.ax_pie.set_visible(True)
                self.ax_pie.set_aspect("equal")
                self.ax_pie.set_position([0.06, 0.10, 0.34, 0.76])
                self.ax_line.set_position([0.50, 0.12, 0.46, 0.76])

                tp, tn, fp, fn = (self.final_tp, self.final_tn,
                                  self.final_fp, self.final_fn)
                total = tp + tn + fp + fn
                vals  = [v for v in [tp, tn, fp, fn] if v > 0]
                lbls  = [l for v, l in zip([tp, tn, fp, fn], ["TP", "TN", "FP", "FN"]) if v > 0]
                clrs  = [c for v, c in zip([tp, tn, fp, fn],
                                           [ACC_C, PREC_C, "#ef4444", REC_C]) if v > 0]
                if vals:
                    _, _, ats = self.ax_pie.pie(
                        vals, labels=lbls, colors=clrs, autopct="%1.1f%%",
                        textprops={"color": TEXT_C, "fontsize": 8.5, "fontweight": "bold"},
                        radius=1,
                        wedgeprops=dict(width=0.46, edgecolor=BG, linewidth=2),
                        startangle=90,
                    )
                    for at in ats:
                        at.set_fontsize(7.5)
                    self.ax_pie.text(0, 0, f"{total}\nitems", ha="center", va="center",
                                     color="#e2e8f0", fontsize=9, weight="bold")
                self.ax_pie.set_title("Prediction Split", color=TITLE_C,
                                      fontsize=10, pad=10, weight="bold")

            # ── Right: history line chart ─────────────────────────────────────
            self.ax_line.clear()
            self.ax_line.set_facecolor(BG)

            if not history:
                self.ax_line.text(
                    0.5, 0.5,
                    "No history yet\nRun a batch analysis to see trends",
                    ha="center", va="center", transform=self.ax_line.transAxes,
                    color="#334155", fontsize=10, style="italic", linespacing=1.8,
                )
                self.ax_line.set_title("Run History \u2014 Metrics Trend",
                                       color=TITLE_C, fontsize=10, pad=10, weight="bold")
            else:
                runs  = list(range(1, len(history) + 1))
                accs  = [h.get("metrics", {}).get("accuracy",  0) for h in history]
                precs = [h.get("metrics", {}).get("precision", 0) for h in history]
                recs  = [h.get("metrics", {}).get("recall",    0) for h in history]
                f1s   = [h.get("metrics", {}).get("f1",        0) for h in history]

                def _plot(ax: Any, x: list, y: list, color: str, label: str) -> None:
                    ax.plot(x, y, color=color, linewidth=2, label=label,
                            marker="o", markersize=5, markerfacecolor=BG,
                            markeredgecolor=color, markeredgewidth=1.8,
                            solid_capstyle="round", solid_joinstyle="round")
                    ax.plot(x, y, color=color, linewidth=6, alpha=0.12, zorder=1)

                _plot(self.ax_line, runs, accs,  ACC_C,  "Accuracy")
                _plot(self.ax_line, runs, precs, PREC_C, "Precision")
                _plot(self.ax_line, runs, recs,  REC_C,  "Recall")
                _plot(self.ax_line, runs, f1s,   F1_C,   "F1 Score")

                self.ax_line.set_xlim(0.5, max(runs) + 0.5)
                self.ax_line.set_ylim(-0.05, 1.08)
                self.ax_line.set_xticks(runs)
                self.ax_line.set_xticklabels(
                    [f"#{r}" for r in runs], color=TEXT_C, fontsize=8,
                )
                self.ax_line.yaxis.set_tick_params(labelcolor=TEXT_C, labelsize=8)
                self.ax_line.set_axisbelow(True)
                self.ax_line.grid(axis="y", color=GRID_C, linewidth=0.8,
                                  linestyle="--", alpha=0.7)
                self.ax_line.grid(axis="x", color=GRID_C, linewidth=0.4,
                                  linestyle=":", alpha=0.4)
                for sp in ("top", "right"):
                    self.ax_line.spines[sp].set_visible(False)
                for sp in ("left", "bottom"):
                    self.ax_line.spines[sp].set_edgecolor(GRID_C)
                    self.ax_line.spines[sp].set_linewidth(0.8)

                leg = self.ax_line.legend(
                    loc="lower right", fontsize=8, facecolor="#161b22",
                    edgecolor=GRID_C, labelcolor=TEXT_C, framealpha=0.9,
                    handlelength=1.6, handletextpad=0.5,
                )
                for line in leg.get_lines():
                    line.set_linewidth(2)

                for val, colour in zip(
                    [accs[-1], precs[-1], recs[-1], f1s[-1]],
                    [ACC_C, PREC_C, REC_C, F1_C],
                ):
                    self.ax_line.annotate(
                        f"{val:.2f}", xy=(runs[-1], val),
                        xytext=(4, 0), textcoords="offset points",
                        color=colour, fontsize=7.5, fontweight="bold", va="center",
                    )

                self.ax_line.set_title("Run History \u2014 Metrics Trend",
                                       color=TITLE_C, fontsize=10, pad=10, weight="bold")

            self.graph_canvas.draw_idle()

    def load_history(self) -> list[dict]:
        if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
            try:
                with open(self.history_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return data if isinstance(data, list) else []
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning(f"Failed to read history: {exc}")
        return []

    def save_to_history(self, record: dict) -> None:
        hist     = self.load_history()
        hist.append(record)
        tmp_path = f"{self.history_file}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(hist, fh, indent=4)
            os.replace(tmp_path, self.history_file)   # atomic on all POSIX platforms
        except OSError as exc:
            logging.warning(f"Failed to save history: {exc}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def show_history_window(self) -> None:
        hist = self.load_history()[-10:]
        if not hist:
            messagebox.showinfo("History", "No history available yet.")
            return

        win = ctk.CTkToplevel(self)
        win.title("Run History \u2014 Last 10 Executions")
        win.geometry("640x520")
        win.attributes("-topmost", True)
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(win, text="Batch Run History",
                     font=_F_HEADING, text_color="#93c5fd").grid(
            row=0, column=0, pady=(14, 6), padx=16, sticky="w",
        )

        textbox = ctk.CTkTextbox(win, font=_F_MONO, text_color="#e2e8f0",
                                  fg_color="#111827", corner_radius=8)
        textbox.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        for i, h in enumerate(reversed(hist)):
            if "time" in h and "date" in h:
                t_display = f"{h['date']} at {h['time']}"
            else:
                t_display = h.get("timestamp", "")[:19]
            m   = h.get("metrics", {})
            idx = len(hist) - i
            textbox.insert("end", f" Run #{idx}  \u2014  {t_display}\n")
            textbox.insert("end",
                f"   Acc:  {m.get('accuracy', 0):.4f}    "
                f"Prec: {m.get('precision', 0):.4f}    "
                f"Rec:  {m.get('recall', 0):.4f}    "
                f"F1:   {m.get('f1', 0):.4f}\n")
            textbox.insert("end",
                f"   TP: {m.get('tp', 0):>4}  TN: {m.get('tn', 0):>4}  "
                f"FP: {m.get('fp', 0):>4}  FN: {m.get('fn', 0):>4}  "
                f"Expert: {m.get('expert_ratio', 0):.1%}\n")
            textbox.insert("end", "   " + "\u2500" * 60 + "\n\n")

        textbox.configure(state="disabled")

    def view_logs(self) -> None:
        """Open the debug log file in a scrollable viewer window."""
        if not os.path.exists(self.log_file):
            messagebox.showinfo("Logs", "No log file found.")
            return

        try:
            # FIX: errors='replace' prevents UnicodeDecodeError from Windows ANSI bytes
            # (e.g. byte 0x85 = next-line character written by some libraries).
            # Also catch UnicodeDecodeError explicitly as belt-and-suspenders since
            # it is NOT a subclass of OSError.
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as fh:
                log_data = fh.read()
        except (OSError, UnicodeDecodeError) as exc:
            messagebox.showerror("Error", f"Could not read logs:\n{exc}")
            return

        win = ctk.CTkToplevel(self)
        win.title("Application Logs")
        win.geometry("900x600")
        win.transient(self)
        win.grab_set()
        win.grid_rowconfigure(1, weight=1)
        win.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(win, text="System Event Logs",
                     font=_F_SUBHEAD, text_color="#f8fafc").grid(
            row=0, column=0, pady=(14, 6), padx=16, sticky="w",
        )

        textbox = ctk.CTkTextbox(win, font=_F_MONO, text_color="#e2e8f0",
                                  fg_color="#111827", corner_radius=8, wrap="word")
        textbox.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        textbox.insert("0.0", log_data)
        textbox.configure(state="disabled")
        textbox.see("end")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    def export_results(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            initialfile=f"result_{timestamp}",
            defaultextension=".pdf",
            filetypes=[
                ("PDF Report",  "*.pdf"),
                ("JSON Data",   "*.json"),
                ("PNG Graphs",  "*.png"),
            ],
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            # FIX: replaced blocking messagebox before thread start with a status update.
            # The old code showed showinfo("Exporting",...) which blocked the main thread
            # while the export thread was running → appeared to hang.
            self.status_label.configure(
                text="Exporting PDF\u2026 please wait.", text_color="#f59e0b",
            )

            def _pdf_task() -> None:
                try:
                    with PdfPages(file_path) as pdf:
                        self._export_combined_pdf_page(pdf)
                    logging.info(f"PDF exported: {file_path}")
                    self.safe_after(0, lambda: [
                        self.status_label.configure(text="PDF exported.", text_color="#22c55e"),
                        messagebox.showinfo("Export Success", f"PDF exported to:\n{file_path}"),
                    ])
                except Exception as exc:
                    logging.error(f"PDF export failed: {exc}")
                    self.safe_after(0, lambda e=exc: [
                        self.status_label.configure(text="Export failed.", text_color="#ef4444"),
                        messagebox.showerror("Export Error", f"Failed to export:\n{e}"),
                    ])

            threading.Thread(target=_pdf_task, daemon=True).start()

        elif ext == ".json":
            try:
                with self._results_lock:
                    export_data = [
                        {k: v for k, v in row.items() if k != "img_path"}
                        for row in self.batch_results_data
                    ]
                with open(file_path, "w", encoding="utf-8") as fh:
                    json.dump(export_data, fh, indent=4, ensure_ascii=False)
                messagebox.showinfo("Export Success", f"JSON exported to:\n{file_path}")
                logging.info(f"JSON exported: {file_path}")
            except Exception as exc:
                logging.error(f"JSON export failed: {exc}")
                messagebox.showerror("Export Error", f"Failed to export:\n{exc}")

        elif ext == ".png":
            stem       = os.path.splitext(os.path.basename(file_path))[0]
            out_folder = os.path.join(os.path.dirname(file_path), f"{stem}_graphs")
            os.makedirs(out_folder, exist_ok=True)

            # FIX: same as PDF — update status bar instead of blocking messagebox
            self.status_label.configure(
                text="Exporting PNGs\u2026 please wait.", text_color="#f59e0b",
            )

            def _png_task() -> None:
                try:
                    self._export_metrics_png(out_folder)
                    logging.info(f"PNG graphs exported: {out_folder}")
                    self.safe_after(0, lambda: [
                        self.status_label.configure(
                            text="PNGs exported.", text_color="#22c55e",
                        ),
                        messagebox.showinfo(
                            "Export Success",
                            f"4 PNG graphs saved to:\n{out_folder}",
                        ),
                    ])
                except Exception as exc:
                    logging.error(f"PNG export failed: {exc}")
                    self.safe_after(0, lambda e=exc: [
                        self.status_label.configure(
                            text="Export failed.", text_color="#ef4444",
                        ),
                        messagebox.showerror("Export Error", f"Failed to export:\n{e}"),
                    ])

            threading.Thread(target=_png_task, daemon=True).start()

        else:
            messagebox.showerror("Export Error", "Unsupported export format selected.")

    def _export_combined_pdf_page(self, pdf: PdfPages) -> None:
        """
        Build multi-page PDF report.
        Page 1  — summary donut + metrics text
        Page 2+ — appendix: 5 items per page with images
        NOTE: Runs on a background thread.  All matplotlib objects here are
              created locally; the shared self.fig/ax_* are never touched.
              _MATPLOTLIB_LOCK is acquired around rendering calls to avoid
              concurrent font-manager access with the main thread.
        """
        import textwrap
        import matplotlib.gridspec as gridspec
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        BG     = "#1a1f2e";  CARD   = "#242b3d"
        ACCENT = "#3b82f6";  WHITE  = "#e2e8f0"
        GRAY   = "#8892a4";  GREEN  = "#22c55e"
        RED    = "#ef4444";  BLUE   = "#3498db"
        YELLOW = "#f39c12"

        json_fname = os.path.basename(getattr(self, "json_path", "") or "Unknown")
        tp, tn, fp, fn = self.final_tp, self.final_tn, self.final_fp, self.final_fn
        total = tp + tn + fp + fn

        # ── Page 1: Summary ───────────────────────────────────────────────────
        fig1 = Figure(figsize=(8.27, 11.69), dpi=150)
        FigureCanvasAgg(fig1)
        fig1.patch.set_facecolor(BG)

        outer = gridspec.GridSpec(
            2, 1, figure=fig1,
            height_ratios=[0.09, 0.88], hspace=0.03,
            left=0.07, right=0.93, top=0.97, bottom=0.03,
        )
        ax_hdr = fig1.add_subplot(outer[0])
        ax_hdr.set_facecolor(BG); ax_hdr.axis("off")
        ax_hdr.axhline(y=0.12, color=ACCENT, linewidth=2)
        ax_hdr.text(0.0, 0.95,
                    "Multimodal Fake News Detector  \u2014  Batch Analysis Report",
                    transform=ax_hdr.transAxes, fontsize=14, fontweight="bold",
                    color=WHITE, va="top", fontfamily="DejaVu Sans")
        ax_hdr.text(0.0, 0.52, f"Source file: {json_fname}",
                    transform=ax_hdr.transAxes, fontsize=8.5, color=ACCENT,
                    va="top", style="italic", fontfamily="DejaVu Sans")
        ax_hdr.text(1.0, 0.95, datetime.now().strftime("%d %b %Y  %H:%M"),
                    transform=ax_hdr.transAxes, fontsize=8.5, color=GRAY,
                    va="top", ha="right", fontfamily="DejaVu Sans")

        body_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1],
            width_ratios=[0.44, 0.56], wspace=0.06,
        )
        ax_pie = fig1.add_subplot(body_gs[0])
        ax_pie.set_facecolor(CARD); ax_pie.set_aspect("equal")

        lbl_all = ["TP", "TN", "FP", "FN"]
        col_all = [GREEN, BLUE, RED, YELLOW]
        val_all = [tp, tn, fp, fn]
        fv = [v for v in val_all if v > 0]
        fl = [l for v, l in zip(val_all, lbl_all) if v > 0]
        fc = [c for v, c in zip(val_all, col_all) if v > 0]

        if fv:
            _, _, autotexts = ax_pie.pie(
                fv, labels=fl, colors=fc, autopct="%1.1f%%",
                textprops={"color": WHITE, "fontsize": 9, "fontweight": "bold"},
                radius=1, wedgeprops=dict(width=0.46, edgecolor=BG, linewidth=1.5),
                startangle=90,
            )
            for at in autotexts:
                at.set_fontsize(8)
        else:
            ax_pie.pie([1], colors=[GRAY], radius=1,
                       wedgeprops=dict(width=0.46, edgecolor=BG))

        ax_pie.text(0, 0, f"{total}\nItems", ha="center", va="center",
                    color=WHITE, fontsize=11, fontweight="bold")
        ax_pie.set_title("Prediction Distribution", color=WHITE,
                         fontsize=10, pad=10, fontweight="bold")
        for sp in ax_pie.spines.values():
            sp.set_visible(False)

        ax_met = fig1.add_subplot(body_gs[1])
        ax_met.set_facecolor(CARD); ax_met.axis("off")
        for sp in ax_met.spines.values():
            sp.set_edgecolor(ACCENT); sp.set_linewidth(0.8); sp.set_visible(True)
        ax_met.text(0.05, 0.97, "Analysis Summary",
                    transform=ax_met.transAxes, fontsize=11, fontweight="bold",
                    color=ACCENT, va="top", fontfamily="DejaVu Sans")
        ax_met.text(0.05, 0.88, self.report_text.strip(),
                    transform=ax_met.transAxes, fontsize=8.5, color=WHITE,
                    va="top", linespacing=1.65, fontfamily="Courier New")

        fig1.text(0.5, 0.008,
                  "Fake News Detection System  \u2022  Multimodal AI Analysis Report",
                  ha="center", fontsize=7, color=GRAY, style="italic",
                  fontfamily="DejaVu Sans")

        with _MATPLOTLIB_LOCK:
            # FIX: removed bbox_inches="tight" — it triggers font-lock in bg thread
            # and can deadlock with the main thread's draw_idle().
            # Layout is handled by explicit GridSpec instead.
            pdf.savefig(fig1, facecolor=BG)

        # FIX: plt.close() fully releases memory; fig.clf() only clears content
        plt.close(fig1)

        # ── Appendix pages ────────────────────────────────────────────────────
        results = list(self.batch_results_data)
        if not results:
            return

        ITEMS_PER_PAGE = 5
        pages      = [results[i:i + ITEMS_PER_PAGE] for i in range(0, len(results), ITEMS_PER_PAGE)]
        total_pgs  = len(pages)

        def _square_crop(pil_img: Image.Image) -> Image.Image:
            w, h = pil_img.size
            m    = min(w, h)
            l    = (w - m) // 2; t = (h - m) // 2
            return pil_img.crop((l, t, l + m, t + m)).resize(
                (96, 96), Image.Resampling.LANCZOS,
            )

        for pg_idx, pg_items in enumerate(pages):
            padded = list(pg_items) + [None] * (ITEMS_PER_PAGE - len(pg_items))

            fig_ap = Figure(figsize=(8.27, 11.69), dpi=90)
            FigureCanvasAgg(fig_ap)
            fig_ap.patch.set_facecolor(BG)

            ap_outer = gridspec.GridSpec(
                ITEMS_PER_PAGE + 1, 1, figure=fig_ap,
                height_ratios=[0.04] + [1.0] * ITEMS_PER_PAGE,
                hspace=0.06, left=0.04, right=0.96, top=0.98, bottom=0.02,
            )

            ax_h = fig_ap.add_subplot(ap_outer[0])
            ax_h.set_facecolor(BG); ax_h.axis("off")
            ax_h.axhline(y=0.15, color=ACCENT, linewidth=1.0)
            ax_h.text(0.0, 0.98,
                      f"Appendix \u2014 Detailed Predictions  (Page {pg_idx+2} of {total_pgs+1})",
                      transform=ax_h.transAxes, fontsize=10, fontweight="bold",
                      color=WHITE, va="top", fontfamily="DejaVu Sans")
            ax_h.text(1.0, 0.98, f"Source: {json_fname}",
                      transform=ax_h.transAxes, fontsize=7.5, color=GRAY,
                      va="top", ha="right", style="italic", fontfamily="DejaVu Sans")

            for row_idx, item in enumerate(padded):
                if item is None:
                    blank = fig_ap.add_subplot(ap_outer[row_idx + 1])
                    blank.set_facecolor(BG); blank.axis("off")
                    continue

                row_gs = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=ap_outer[row_idx + 1],
                    width_ratios=[0.80, 0.20], wspace=0.02,
                )
                ax_row = fig_ap.add_subplot(row_gs[0])
                ax_row.set_facecolor(CARD); ax_row.axis("off")
                for sp in ax_row.spines.values():
                    sp.set_edgecolor(ACCENT); sp.set_linewidth(0.6); sp.set_visible(True)

                pred_lbl = item.get("predicted_label", "?")
                true_lbl = item.get("true_label", "?")
                conf     = item.get("confidence", 0)
                is_expert = item.get("expert_review", False)
                item_id  = item.get("id", "N/A")
                headline = "\n".join(textwrap.wrap(item.get("text", "")[:220], width=70))

                correct    = pred_lbl == true_lbl
                sym_col    = GREEN if correct else RED
                pred_col   = RED   if pred_lbl == "Fake" else GREEN
                exp_col    = YELLOW if is_expert else GRAY
                PAD        = 0.018

                ax_row.axhline(y=0.96, xmin=PAD, xmax=0.98,
                               color=ACCENT, linewidth=0.4, alpha=0.5)
                ax_row.text(PAD, 0.985, f"ID:  {item_id}",
                            transform=ax_row.transAxes, fontsize=8.5,
                            fontweight="bold", color=WHITE, va="top",
                            fontfamily="DejaVu Sans")
                ax_row.text(0.985, 0.985, "\u2713" if correct else "\u2717",
                            transform=ax_row.transAxes, fontsize=10.5,
                            fontweight="bold", color=sym_col, va="top", ha="right")

                ax_row.text(PAD, 0.89, "News:",
                            transform=ax_row.transAxes, fontsize=7.5,
                            color=GRAY, fontweight="bold", va="top",
                            fontfamily="DejaVu Sans")
                ax_row.text(PAD + 0.07, 0.89, headline,
                            transform=ax_row.transAxes, fontsize=7.5,
                            color="#cbd5e1", va="top", linespacing=1.3,
                            style="italic", fontfamily="DejaVu Sans")

                fields = [
                    (0.41, "Predicted :", f"{pred_lbl:>8}", pred_col),
                    (0.30, "True Label:", f"{true_lbl:>8}", WHITE),
                    (0.19, "Confidence:", f"{conf:.1%}",   ACCENT),
                    (0.08, "Expert Rev:", "Yes [!]" if is_expert else "No", exp_col),
                ]
                for y_f, lbl, val, vcol in fields:
                    ax_row.axhline(y=y_f + 0.08, xmin=PAD, xmax=0.97,
                                   color=GRAY, linewidth=0.25, alpha=0.25)
                    ax_row.text(PAD, y_f, lbl, transform=ax_row.transAxes,
                                fontsize=7.5, color=GRAY, fontweight="bold",
                                va="top", fontfamily="Courier New")
                    ax_row.text(PAD + 0.17, y_f, val, transform=ax_row.transAxes,
                                fontsize=8.0, color=vcol, fontweight="bold",
                                va="top", fontfamily="Courier New")

                ax_im = fig_ap.add_subplot(row_gs[1])
                ax_im.set_facecolor(CARD)
                ax_im.set_xlim(0, 1); ax_im.set_ylim(0, 1)
                ax_im.set_xticks([]); ax_im.set_yticks([])
                for sp in ax_im.spines.values():
                    sp.set_edgecolor(ACCENT); sp.set_linewidth(0.5); sp.set_visible(True)

                M = 0.04
                local_img = item.get("img_path")
                if local_img and os.path.isfile(local_img):
                    try:
                        pil = _square_crop(Image.open(local_img).convert("RGB"))
                        ax_im.imshow(pil, extent=[M, 1-M, M, 1-M],
                                     aspect="equal", origin="upper",
                                     interpolation="bilinear")
                    except Exception:
                        ax_im.text(0.5, 0.5, "Error", ha="center", va="center",
                                   color=GRAY, fontsize=7)
                else:
                    ax_im.text(0.5, 0.5, "No\nImage", ha="center", va="center",
                               color=GRAY, fontsize=7, style="italic", linespacing=1.4)

            with _MATPLOTLIB_LOCK:
                # FIX: removed bbox_inches="tight" to prevent matplotlib font-lock deadlock
                pdf.savefig(fig_ap, facecolor=BG)

            plt.close(fig_ap)   # FIX: was fig_ap.clf() — clf() clears but doesn't free

    def _export_metrics_png(self, out_folder: str) -> None:
        """Save 4 standalone PNG charts to out_folder."""
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        BG     = "#1a1f2e";  CARD   = "#242b3d"
        ACCENT = "#3b82f6";  WHITE  = "#e2e8f0"
        GRAY   = "#8892a4";  GREEN  = "#22c55e"
        RED    = "#ef4444";  BLUE   = "#3498db"
        YELLOW = "#f39c12"

        tp, tn, fp, fn = self.final_tp, self.final_tn, self.final_fp, self.final_fn
        total  = tp + tn + fp + fn
        acc    = (tp + tn) / total      if total       > 0 else 0.0
        prec   = tp / (tp + fp)         if (tp + fp)  > 0 else 0.0
        rec    = tp / (tp + fn)         if (tp + fn)  > 0 else 0.0
        f1     = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        expert = self.final_expert      # FIX: was getattr(self, "final_expert", 0) — now stored

        def _save(fig: Figure, name: str) -> None:
            with _MATPLOTLIB_LOCK:
                # FIX: removed bbox_inches="tight" to prevent font-lock deadlock
                fig.savefig(os.path.join(out_folder, name), facecolor=BG, dpi=150)
            plt.close(fig)              # FIX: was missing — figures leaked memory

        # ── 1. Donut ──────────────────────────────────────────────────────────
        fig1 = Figure(figsize=(6, 6), dpi=150)
        FigureCanvasAgg(fig1)
        ax = fig1.add_subplot(111)
        fig1.patch.set_facecolor(BG); ax.set_facecolor(CARD); ax.set_aspect("equal")
        lbl_all = ["TP", "TN", "FP", "FN"];  col_all = [GREEN, BLUE, RED, YELLOW]
        val_all = [tp, tn, fp, fn]
        fv = [v for v in val_all if v > 0]
        fl = [l for v, l in zip(val_all, lbl_all) if v > 0]
        fc = [c for v, c in zip(val_all, col_all) if v > 0]
        if fv:
            _, _, ats = ax.pie(fv, labels=fl, colors=fc, autopct="%1.1f%%",
                textprops={"color": WHITE, "fontsize": 10, "fontweight": "bold"},
                radius=1, wedgeprops=dict(width=0.48, edgecolor=BG, linewidth=2),
                startangle=90)
            for at in ats: at.set_fontsize(9)
        else:
            ax.pie([1], colors=[GRAY], radius=1,
                   wedgeprops=dict(width=0.48, edgecolor=BG))
        ax.text(0, 0, f"{total}\nItems", ha="center", va="center",
                color=WHITE, fontsize=12, fontweight="bold")
        ax.set_title("Prediction Distribution", color=WHITE, fontsize=12,
                     pad=12, fontweight="bold")
        _save(fig1, "1_pie.png")

        # ── 2. Confusion matrix ───────────────────────────────────────────────
        fig2 = Figure(figsize=(5, 5), dpi=150)
        FigureCanvasAgg(fig2)
        ax = fig2.add_subplot(111)
        fig2.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, int(v), ha="center", va="center", fontsize=16,
                    fontweight="bold",
                    color=WHITE if cm[r, c] > cm.max() * 0.4 else GRAY)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Real", "Pred Fake"], color=WHITE, fontsize=10)
        ax.set_yticklabels(["Act Real", "Act Fake"], color=WHITE, fontsize=10,
                           rotation=90, va="center")
        ax.tick_params(colors=WHITE, length=0)
        for sp in ax.spines.values():
            sp.set_edgecolor(ACCENT); sp.set_linewidth(1)
        ax.set_title("Confusion Matrix", color=WHITE, fontsize=12,
                     pad=10, fontweight="bold")
        _save(fig2, "2_confusion.png")

        # ── 3. Metrics bar ────────────────────────────────────────────────────
        fig3 = Figure(figsize=(7, 4), dpi=150)
        FigureCanvasAgg(fig3)
        ax = fig3.add_subplot(111)
        fig3.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values = [acc, prec, rec, f1]
        bcolors = [BLUE, GREEN, YELLOW, ACCENT]
        bars = ax.barh(names, values, color=bcolors, height=0.5, edgecolor=BG)
        for bar, val in zip(bars, values):
            ax.text(min(val + 0.02, 0.96), bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10,
                    fontweight="bold", color=WHITE)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                           color=GRAY, fontsize=9)
        ax.tick_params(colors=WHITE, length=0)
        ax.yaxis.set_tick_params(labelcolor=WHITE, labelsize=11)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_edgecolor(ACCENT)
        ax.axvline(0.5, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title("Classification Metrics", color=WHITE, fontsize=12,
                     pad=10, fontweight="bold")
        _save(fig3, "3_metrics_bar.png")

        # ── 4. KPI scorecard ──────────────────────────────────────────────────
        fig4 = Figure(figsize=(10, 3), dpi=150)
        FigureCanvasAgg(fig4)
        ax = fig4.add_subplot(111)
        fig4.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
        kpis = [
            ("True Positives",  tp,    GREEN),
            ("True Negatives",  tn,    BLUE),
            ("False Positives", fp,    RED),
            ("False Negatives", fn,    YELLOW),
            ("Total Items",     total, WHITE),
            ("Expert Reviews",  expert, ACCENT),
        ]
        tile_w = 0.148; tile_h = 0.72; pad_x = 0.012
        for i, (lbl, val, col) in enumerate(kpis):
            x = 0.01 + i * (tile_w + pad_x)
            ax.add_patch(mpatches.Rectangle(
                (x, 0.08), tile_w, tile_h,
                transform=ax.transAxes, facecolor=CARD,
                edgecolor=col, linewidth=1.5, clip_on=False,
            ))
            ax.text(x + tile_w / 2, 0.60, str(val),
                    transform=ax.transAxes, fontsize=20,
                    fontweight="bold", color=col, ha="center", va="center")
            ax.text(x + tile_w / 2, 0.20, lbl,
                    transform=ax.transAxes, fontsize=8,
                    color=GRAY, ha="center", va="center")
        ax.set_title("KPI Scorecard", color=WHITE, fontsize=12,
                     pad=8, fontweight="bold")
        _save(fig4, "4_kpi_scorecard.png")


if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()