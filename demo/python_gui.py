"""
python_gui.py — Indian Traffic Detection System
================================================
Features:
  • Load image or video file for detection
  • Live webcam detection
  • Direct YOLO inference (no subprocess overhead)
  • Falls back to YOLOv8n-COCO if custom model isn't trained yet

Run from project root:
  python demo/python_gui.py
"""

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Model paths ───────────────────────────────────────────────────────────────
CUSTOM_MODEL   = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")
FALLBACK_MODEL = os.path.join(ROOT, "yolov8n.pt")

# COCO vehicle classes (fallback mode)
COCO_TO_INDIAN = {
    0: "pedestrian", 1: "bicycle", 2: "car",
    3: "two_wheeler", 5: "bus", 7: "truck",
}

# ── Colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#0d0f1a"
BG_MID    = "#141726"
BG_CARD   = "#1c2035"
ACCENT    = "#00e5ff"
ACCENT2   = "#7c4dff"
TEXT      = "#e0e6f0"
TEXT_DIM  = "#7a8299"
BTN_BG    = "#1e2540"
BTN_HOVER = "#2a3260"
SUCCESS   = "#00e676"
WARNING   = "#ffab40"
ERROR     = "#ff1744"
WEBCAM_CLR = "#ff6d00"

BOX_COLORS = [
    (0, 229, 255), (124, 77, 255), (0, 230, 118),
    (255, 214, 0), (255, 64, 129), (255, 111, 0),
    (0, 200, 83),  (213, 0, 249), (29, 233, 182),
    (255, 196, 0),
]

FONT_LABEL  = ("Segoe UI", 10, "bold")
FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_STATUS = ("Consolas", 10)
FONT_BTN    = ("Segoe UI", 10, "bold")
FONT_HEADER = ("Segoe UI", 17, "bold")


class TrafficGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🚦 Indian Traffic Detection System")
        self.geometry("1600x820")
        self.minsize(1200, 680)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── state ─────────────────────────────────────────────────────────────
        self.yolo_model      = None
        self.model_is_custom = False
        self.model_loading   = True

        self.current_image_path = None
        self.current_image_cv   = None
        self.annotated_image    = None
        self.results            = None

        self.video_cap    = None
        self.video_path   = None

        self.webcam_active   = False
        self.webcam_cap      = None
        self.webcam_queue    = queue.Queue(maxsize=3)
        self._frame_counter  = 0

        self._build_ui()
        self._set_status("Loading YOLO model…  ⌛", TEXT_DIM)

        # Load YOLO in background so UI stays responsive
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(14, 2))

        tk.Label(hdr, text="🚦  Indian Traffic Detection",
                 font=FONT_HEADER, fg=ACCENT, bg=BG_DARK).pack(side="left")
        self.model_lbl = tk.Label(hdr, text="[ loading model… ]",
                                  font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_DARK)
        self.model_lbl.pack(side="left", padx=14)

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=20, pady=4)

        # Main content row: image panels + results sidebar
        content_row = tk.Frame(self, bg=BG_DARK)
        content_row.pack(fill="both", expand=True, padx=20, pady=4)

        # Left: two image panels
        canvas_row = tk.Frame(content_row, bg=BG_DARK)
        canvas_row.pack(side="left", fill="both", expand=True)

        self.orig_panel  = self._make_panel(canvas_row, "📷  Original Frame")
        self.annot_panel = self._make_panel(canvas_row, "🔍  Detected Vehicles")

        # Right: detection results sidebar
        self._build_results_sidebar(content_row)

        # Button bar
        btn_row = tk.Frame(self, bg=BG_MID, pady=10)
        btn_row.pack(fill="x")

        self.btn_load   = self._btn(btn_row, "📂  Load File",    self._load_file)
        self.btn_detect = self._btn(btn_row, "▶  Detect",        self._run_detection)
        self.btn_prev   = self._btn(btn_row, "⏮  Prev",          self._prev_frame)
        self.btn_next   = self._btn(btn_row, "⏭  Next",          self._next_frame)
        self.btn_save   = self._btn(btn_row, "💾  Save Result",   self._save_result)
        self.btn_webcam = self._btn(btn_row, "📷  Webcam",        self._toggle_webcam,
                                    accent=WEBCAM_CLR)

        # Confidence slider
        conf_frame = tk.Frame(btn_row, bg=BG_MID)
        conf_frame.pack(side="right", padx=16)
        tk.Label(conf_frame, text="Confidence:", font=("Segoe UI", 9),
                 fg=TEXT_DIM, bg=BG_MID).pack(side="left", padx=(0, 4))
        self.conf_var = tk.DoubleVar(value=0.15)
        self.conf_slider = tk.Scale(
            conf_frame, variable=self.conf_var,
            from_=0.05, to=0.80, resolution=0.05,
            orient="horizontal", length=130,
            bg=BG_MID, fg=ACCENT, troughcolor=BG_CARD,
            highlightthickness=0, bd=0,
            font=("Segoe UI", 8), showvalue=True)
        self.conf_slider.pack(side="left")

        self.btn_detect.config(state="disabled")
        self.btn_prev.config(state="disabled")
        self.btn_next.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.btn_webcam.config(state="disabled")

        for b in (self.btn_load, self.btn_detect, self.btn_prev,
                  self.btn_next, self.btn_save, self.btn_webcam):
            b.pack(side="left", padx=8, pady=4)

        # Status bar
        sb = tk.Frame(self, bg=BG_CARD, height=26)
        sb.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar()
        self.status_lbl = tk.Label(sb, textvariable=self.status_var,
                                   font=FONT_STATUS, fg=TEXT_DIM, bg=BG_CARD,
                                   anchor="w", padx=12)
        self.status_lbl.pack(fill="x")

    def _make_panel(self, parent, title: str) -> tk.Label:
        frame = tk.Frame(parent, bg=BG_CARD, bd=0,
                         highlightbackground=ACCENT2, highlightthickness=1)
        frame.pack(side="left", fill="both", expand=True, padx=6)
        tk.Label(frame, text=title, font=FONT_TITLE,
                 fg=ACCENT, bg=BG_CARD).pack(pady=(8, 2))
        canvas = tk.Label(frame, bg="#0a0c16", cursor="crosshair")
        canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        return canvas

    def _build_results_sidebar(self, parent):
        """Right-side panel showing detection results as a readable list."""
        sidebar = tk.Frame(parent, bg=BG_CARD, bd=0,
                           highlightbackground=ACCENT2, highlightthickness=1,
                           width=260)
        sidebar.pack(side="right", fill="y", padx=(6, 0))
        sidebar.pack_propagate(False)

        # Title
        tk.Label(sidebar, text="🎯  Detection Results",
                 font=FONT_TITLE, fg=ACCENT, bg=BG_CARD).pack(pady=(10, 2))
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=10, pady=4)

        # Summary badge row
        badge_row = tk.Frame(sidebar, bg=BG_CARD)
        badge_row.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(badge_row, text="Total:", font=("Segoe UI", 10),
                 fg=TEXT_DIM, bg=BG_CARD).pack(side="left")
        self.total_badge = tk.Label(badge_row, text="—",
                                    font=("Segoe UI", 12, "bold"),
                                    fg=ACCENT, bg=BG_CARD)
        self.total_badge.pack(side="left", padx=6)

        # Scrollable listbox area
        list_frame = tk.Frame(sidebar, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        scrollbar = tk.Scrollbar(list_frame, orient="vertical",
                                  bg=BG_MID, troughcolor=BG_DARK,
                                  activebackground=ACCENT2)
        scrollbar.pack(side="right", fill="y")

        self.results_listbox = tk.Listbox(
            list_frame,
            font=("Consolas", 10),
            fg=TEXT,
            bg="#0d1120",
            selectbackground=ACCENT2,
            selectforeground="#ffffff",
            activestyle="none",
            relief="flat",
            bd=0,
            yscrollcommand=scrollbar.set
        )
        self.results_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.results_listbox.yview)

        # Per-class counts section
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=10, pady=4)
        tk.Label(sidebar, text="📊  Class Summary",
                 font=("Segoe UI", 10, "bold"), fg=ACCENT2, bg=BG_CARD).pack(pady=(0, 4))
        self.class_summary_frame = tk.Frame(sidebar, bg=BG_CARD)
        self.class_summary_frame.pack(fill="x", padx=10, pady=(0, 10))

    def _update_results_panel(self, classes: list, confs: list):
        """Populate the results sidebar with detection data."""
        self.results_listbox.delete(0, tk.END)
        for widget in self.class_summary_frame.winfo_children():
            widget.destroy()

        if not classes:
            self.total_badge.config(text="0", fg=TEXT_DIM)
            self.results_listbox.insert(tk.END, "  No objects detected.")
            self.results_listbox.itemconfig(0, fg=TEXT_DIM)
            return

        self.total_badge.config(text=str(len(classes)), fg=ACCENT)

        # Detailed list
        for idx, (cls, conf) in enumerate(zip(classes, confs)):
            bar_len = int(conf * 14)
            bar = "█" * bar_len + "░" * (14 - bar_len)
            line = f"  #{idx+1:<2}  {cls:<15}  {conf:.0%}  {bar}"
            self.results_listbox.insert(tk.END, line)
            col_idx = idx % len(BOX_COLORS)
            r, g, b = BOX_COLORS[col_idx]
            hex_col = f"#{r:02x}{g:02x}{b:02x}"
            self.results_listbox.itemconfig(idx, fg=hex_col)

        # Class count summary
        from collections import Counter
        counts = Counter(classes)
        for cls_name, count in sorted(counts.items(), key=lambda x: -x[1]):
            row = tk.Frame(self.class_summary_frame, bg=BG_CARD)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"  {cls_name}",
                     font=("Segoe UI", 9), fg=TEXT, bg=BG_CARD,
                     anchor="w").pack(side="left", fill="x", expand=True)
            tk.Label(row, text=f"×{count}",
                     font=("Segoe UI", 9, "bold"), fg=WARNING, bg=BG_CARD,
                     width=4).pack(side="right")

    def _btn(self, parent, text: str, cmd, accent: str = ACCENT) -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd,
                      font=FONT_BTN, fg=accent, bg=BTN_BG,
                      activebackground=BTN_HOVER, activeforeground=accent,
                      relief="flat", bd=0, padx=14, pady=7, cursor="hand2")
        b.bind("<Enter>", lambda e: b.config(bg=BTN_HOVER))
        b.bind("<Leave>", lambda e: b.config(bg=BTN_BG))
        return b

    # ── helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = TEXT):
        self.status_var.set(f"  {msg}")
        self.status_lbl.config(fg=color)
        self.update_idletasks()

    def _display_cv(self, panel: tk.Label, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        panel.update_idletasks()
        w = max(panel.winfo_width(),  400)
        h = max(panel.winfo_height(), 300)
        pil.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        panel.config(image=photo)
        panel.image = photo

    def _display_pil(self, panel: tk.Label, pil_img: Image.Image):
        panel.update_idletasks()
        w = max(panel.winfo_width(),  400)
        h = max(panel.winfo_height(), 300)
        img = pil_img.copy()
        img.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        panel.config(image=photo)
        panel.image = photo

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            if os.path.exists(CUSTOM_MODEL):
                mdl = YOLO(CUSTOM_MODEL)
                custom = True
                name   = "indian_vehicles_yolo.pt"
            else:
                mdl  = YOLO(FALLBACK_MODEL)
                custom = False
                name   = "yolov8n.pt (COCO fallback)"
            self.yolo_model      = mdl
            self.model_is_custom = custom
            self.after(0, lambda: self._on_model_ready(name))
        except Exception as exc:
            self.after(0, lambda: self._set_status(f"Model load error: {exc}", ERROR))

    def _on_model_ready(self, model_name: str):
        self.model_loading = False
        self.model_lbl.config(text=f"[ {model_name} ]", fg=SUCCESS)
        self.btn_load.config(state="normal")
        self.btn_webcam.config(state="normal")
        self._set_status(f"✅  Ready — {model_name}", SUCCESS)

    # ── file loading ──────────────────────────────────────────────────────────

    def _load_file(self):
        if self.webcam_active:
            self._stop_webcam()

        path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[("Image / Video",
                        "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            self._open_video(path)
        else:
            self._open_image(path)

        self.btn_detect.config(state="normal")
        self.btn_save.config(state="disabled")
        self.annotated_image = None
        self.annot_panel.config(image="")
        self.annot_panel.image = None

    def _open_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            self._set_status("Could not read image.", ERROR)
            return
        self.current_image_cv   = img
        self.current_image_path = path
        self.video_cap = None
        self._display_cv(self.orig_panel, img)
        self.btn_prev.config(state="disabled")
        self.btn_next.config(state="disabled")
        self._set_status(f"Image loaded: {os.path.basename(path)}", SUCCESS)

    def _open_video(self, path: str):
        if self.video_cap:
            self.video_cap.release()
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok:
            self._set_status("Could not read video.", ERROR)
            return
        self.video_cap  = cap
        self.video_path = path
        self.current_image_cv   = frame
        self.current_image_path = self._save_temp(frame)
        self._display_cv(self.orig_panel, frame)
        self.btn_prev.config(state="normal")
        self.btn_next.config(state="normal")
        self._set_status(f"Video: {os.path.basename(path)}  [{total} frames]", SUCCESS)

    def _prev_frame(self):
        if not self.video_cap:
            return
        pos = max(0, int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = self.video_cap.read()
        if ok:
            self.current_image_cv   = frame
            self.current_image_path = self._save_temp(frame)
            self._display_cv(self.orig_panel, frame)
            self._set_status(f"Frame {pos + 1}", TEXT_DIM)

    def _next_frame(self):
        if not self.video_cap:
            return
        ok, frame = self.video_cap.read()
        if ok:
            pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.current_image_cv   = frame
            self.current_image_path = self._save_temp(frame)
            self._display_cv(self.orig_panel, frame)
            self._set_status(f"Frame {pos}", TEXT_DIM)

    def _save_temp(self, frame) -> str:
        tmp = os.path.join(ROOT, "_temp_gui_frame.jpg")
        cv2.imwrite(tmp, frame)
        return tmp

    # ── single-image detection ────────────────────────────────────────────────

    def _run_detection(self):
        if self.current_image_path is None:
            self._set_status("Load an image or video first!", ERROR)
            return
        if self.model_loading or self.yolo_model is None:
            self._set_status("Model is still loading…", WARNING)
            return
        self._set_status("Running detection…  ⏳", ACCENT)
        self.btn_detect.config(state="disabled")
        threading.Thread(target=self._detect_worker, daemon=True).start()

    def _detect_worker(self):
        try:
            img = self.current_image_cv
            conf_thresh = self.conf_var.get()
            results_raw = self.yolo_model(
                self.current_image_path, conf=conf_thresh, verbose=False)[0]
            bboxes, classes, confs = self._parse_results(results_raw)

            # Annotate
            annotated = self._draw_pil(img, bboxes, classes, confs)
            self.annotated_image = annotated
            self.results = {"bboxes": bboxes, "classes": classes, "confs": confs}
            self.after(0, self._post_detect, len(bboxes))
        except Exception as exc:
            self.after(0, lambda: self._detection_error(str(exc)))

    def _post_detect(self, count: int):
        self._display_pil(self.annot_panel, self.annotated_image)
        cls  = self.results.get("classes", []) if self.results else []
        conf = self.results.get("confs",   []) if self.results else []
        self._update_results_panel(cls, conf)
        self._set_status(f"✅  {count} vehicle(s) detected.", SUCCESS)
        self.btn_detect.config(state="normal")
        self.btn_save.config(state="normal")

    def _detection_error(self, msg: str):
        self._set_status(f"Error: {msg}", ERROR)
        self.btn_detect.config(state="normal")
        messagebox.showerror("Detection Failed", msg)

    # ── webcam ────────────────────────────────────────────────────────────────

    def _toggle_webcam(self):
        if self.webcam_active:
            self._stop_webcam()
        else:
            self._start_webcam()

    def _start_webcam(self):
        if self.yolo_model is None:
            self._set_status("Model not loaded yet.", WARNING)
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._set_status("❌  Cannot open webcam (index 0).", ERROR)
            return

        self.webcam_cap    = cap
        self.webcam_active = True
        self._frame_counter = 0

        # Clear both panels
        self.orig_panel.config(image="");  self.orig_panel.image  = None
        self.annot_panel.config(image=""); self.annot_panel.image = None
        self.btn_webcam.config(text="⏹  Stop Webcam", fg=ERROR)
        self.btn_detect.config(state="disabled")
        self.btn_load.config(state="disabled")
        self._set_status("🎥  Webcam active — detecting live…", WEBCAM_CLR)

        # Worker thread: capture + YOLO
        threading.Thread(target=self._webcam_worker, daemon=True).start()
        # Display poll: main-thread safe
        self._webcam_poll()

    def _stop_webcam(self):
        self.webcam_active = False
        time.sleep(0.15)                 # let worker thread exit
        if self.webcam_cap:
            self.webcam_cap.release()
            self.webcam_cap = None
        with self.webcam_queue.mutex:
            self.webcam_queue.queue.clear()

        self.btn_webcam.config(text="📷  Webcam", fg=WEBCAM_CLR)
        self.btn_detect.config(state="normal")
        self.btn_load.config(state="normal")
        self._set_status("Webcam stopped.", TEXT_DIM)

    def _webcam_worker(self):
        """Background thread: reads frames and runs YOLO every N frames."""
        last_boxes, last_classes, last_confs = [], [], []
        detect_every = 4          # run YOLO on every 4th captured frame
        t_last = time.time()

        while self.webcam_active:
            ok, frame = self.webcam_cap.read()
            if not ok:
                break

            self._frame_counter += 1

            # Throttle: detect every N frames
            if self._frame_counter % detect_every == 0:
                try:
                    conf_thresh = self.conf_var.get()
                    res = self.yolo_model(frame, conf=conf_thresh, verbose=False,
                                          imgsz=416)[0]      # 416 = faster on CPU
                    last_boxes, last_classes, last_confs = self._parse_results(res)
                except Exception:
                    pass

            # Draw bboxes on frame
            annotated = self._draw_cv(frame.copy(), last_boxes, last_classes, last_confs)

            # Push to queue (drop old frame if full)
            payload = (frame.copy(), annotated, len(last_boxes),
                       list(last_classes), list(last_confs))
            try:
                self.webcam_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self.webcam_queue.get_nowait()
                    self.webcam_queue.put_nowait(payload)
                except queue.Empty:
                    pass

            # ~15 FPS capture cap
            elapsed = time.time() - t_last
            sleep   = max(0, (1/15) - elapsed)
            time.sleep(sleep)
            t_last = time.time()

    def _webcam_poll(self):
        """Main-thread poll — reads from queue and updates display."""
        if not self.webcam_active:
            return
        try:
            frame, annotated, count, cls, conf = self.webcam_queue.get_nowait()
            self._display_cv(self.orig_panel,   frame)
            self._display_cv(self.annot_panel,  annotated)
            self._update_results_panel(cls, conf)
            mode = "🟢 Custom model" if self.model_is_custom else "⚪ COCO fallback"
            self._set_status(
                f"🎥 Live  |  {count} vehicle(s)  |  {mode}", WEBCAM_CLR)
        except queue.Empty:
            pass
        # Re-schedule
        self.after(66, self._webcam_poll)    # ~15 FPS

    # ── result parsing ────────────────────────────────────────────────────────

    def _parse_results(self, results_raw):
        bboxes, classes, confs = [], [], []
        for box in results_raw.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            if self.model_is_custom:
                name = results_raw.names.get(cls_id, f"class_{cls_id}")
            else:
                if cls_id not in COCO_TO_INDIAN:
                    continue
                name = COCO_TO_INDIAN[cls_id]

            bboxes.append([x1, y1, x2, y2])
            classes.append(name)
            confs.append(round(conf, 3))
        return bboxes, classes, confs

    # ── drawing (PIL — for saved images) ─────────────────────────────────────

    def _draw_pil(self, cv_img, bboxes, classes, confs) -> Image.Image:
        rgb  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arialbd.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for i, (bbox, cls, conf) in enumerate(zip(bboxes, classes, confs)):
            x1, y1, x2, y2 = bbox
            col = BOX_COLORS[i % len(BOX_COLORS)]
            for t in range(3):
                draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=col)
            label = f"{cls}  {conf:.2f}"
            tb = draw.textbbox((x1, max(y1 - 22, 0)), label, font=font)
            draw.rectangle(tb, fill=col)
            draw.text((x1, max(y1 - 22, 0)), label, fill=(10, 12, 26), font=font)
        return pil

    # ── drawing (cv2 — fast, for webcam) ─────────────────────────────────────

    def _draw_cv(self, frame, bboxes, classes, confs):
        for i, (bbox, cls, conf) in enumerate(zip(bboxes, classes, confs)):
            x1, y1, x2, y2 = bbox
            col = BOX_COLORS[i % len(BOX_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            label = f"{cls} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            bg_y = max(y1 - 22, 0)
            cv2.rectangle(frame, (x1, bg_y), (x1 + tw + 4, bg_y + th + 6), col, -1)
            cv2.putText(frame, label, (x1 + 2, bg_y + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 12, 26), 1, cv2.LINE_AA)
        return frame

    # ── save ─────────────────────────────────────────────────────────────────

    def _save_result(self):
        if self.annotated_image is None:
            self._set_status("Run detection first!", ERROR)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="Save Annotated Image")
        if not path:
            return
        self.annotated_image.save(path)
        self._set_status(f"Saved → {os.path.basename(path)}", SUCCESS)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        self.webcam_active = False
        if self.webcam_cap:
            self.webcam_cap.release()
        if self.video_cap:
            self.video_cap.release()
        self.destroy()


if __name__ == "__main__":
    os.chdir(ROOT)
    app = TrafficGUI()
    app.mainloop()
