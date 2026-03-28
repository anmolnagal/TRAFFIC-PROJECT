"""
label_reviewer.py
=================
A lightweight label review tool built with Tkinter + OpenCV.
No external annotation tool needed — works with your existing setup.

Usage (from project root):
    python src/label_reviewer.py

Controls:
    ← / → arrows  or  A / D    Previous / Next image
    Delete / Backspace          Remove the SELECTED box (click to select)
    Ctrl+S  or  S               Save current labels
    Q                           Quit (will prompt to save unsaved changes)

Click on a bounding box to select it. Click in empty space to deselect.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR    = os.path.join(ROOT, "data", "custom_congestion", "images")
LBL_DIR    = os.path.join(ROOT, "data", "custom_congestion", "labels")

CLASS_NAMES = ["auto", "bicycle", "bus", "car", "motorcycle",
               "pedestrian", "tempo", "tractor", "truck", "van"]

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

BOX_COLORS = [
    (0, 229, 255), (124, 77, 255), (0, 230, 118), (255, 214, 0),
    (255, 64, 129), (255, 111, 0), (0, 200, 83), (213, 0, 249),
    (29, 233, 182), (255, 196, 0),
]

# Dark theme colours
BG       = "#0d0f1a"
BG_MID   = "#141726"
BG_CARD  = "#1c2035"
ACCENT   = "#00e5ff"
ACCENT2  = "#7c4dff"
TEXT     = "#e0e6f0"
TEXT_DIM = "#7a8299"
SUCCESS  = "#00e676"
WARNING  = "#ffab40"
ERROR    = "#ff1744"


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _read_label(lbl_path):
    """Returns list of [cls_id, cx, cy, bw, bh] (all floats)."""
    if not os.path.exists(lbl_path):
        return []
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    boxes.append([int(parts[0])] + [float(p) for p in parts[1:]])
                except ValueError:
                    pass
    return boxes


def _write_label(lbl_path, boxes):
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    with open(lbl_path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def _yolo_to_pixel(box, W, H):
    """[cls, cx, cy, bw, bh] → [cls, x1, y1, x2, y2] in pixels."""
    cls_id, cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return [cls_id, x1, y1, x2, y2]


def _pixel_to_yolo(cls_id, x1, y1, x2, y2, W, H):
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return [cls_id, round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)]


class LabelReviewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🏷️  Label Reviewer — Indian Traffic Project")
        self.geometry("1300x800")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        # ── state ─────────────────────────────────────────────────────────────
        self.images      = sorted([
            f for f in os.listdir(IMG_DIR)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ])
        if not self.images:
            messagebox.showerror(
                "No images",
                f"No images found in:\n{IMG_DIR}\n\nRun auto-labeling first.")
            self.destroy()
            return

        self.idx         = 0
        self.boxes       = []      # list of [cls_id, cx, cy, bw, bh]
        self.pixel_boxes = []      # list of [cls_id, x1, y1, x2, y2] scaled to display
        self.selected    = -1      # index of selected box
        self.dirty       = False   # unsaved changes?

        self.orig_w = 1
        self.orig_h = 1
        self.disp_scale = 1.0     # ratio: display pixels / original pixels

        # Drawing new box state
        self._drawing   = False
        self._draw_start = None
        self._draw_rect  = None

        self._build_ui()
        self._load_image(0)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=16, pady=(10, 2))
        tk.Label(hdr, text="🏷️  Label Reviewer", font=("Segoe UI", 15, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        self.counter_lbl = tk.Label(hdr, text="", font=("Segoe UI", 10),
                                    fg=TEXT_DIM, bg=BG)
        self.counter_lbl.pack(side="left", padx=12)
        self.dirty_lbl = tk.Label(hdr, text="", font=("Segoe UI", 9, "bold"),
                                  fg=WARNING, bg=BG)
        self.dirty_lbl.pack(side="left")

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=16, pady=4)

        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=4)

        # ── left: canvas ──────────────────────────────────────────────────────
        canvas_frame = tk.Frame(main, bg=BG_CARD, highlightbackground=ACCENT2,
                                highlightthickness=1)
        canvas_frame.pack(side="left", fill="both", expand=True)

        tk.Label(canvas_frame, text="Click a box to select it  |  "
                 "Draw new box by clicking + dragging",
                 font=("Segoe UI", 8), fg=TEXT_DIM, bg=BG_CARD).pack(pady=(6, 2))

        self.canvas = tk.Canvas(canvas_frame, bg="#0a0c16",
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.canvas.bind("<Configure>",      self._on_resize)
        self.canvas.bind("<Button-1>",       self._on_click)
        self.canvas.bind("<B1-Motion>",      self._on_drag)
        self.canvas.bind("<ButtonRelease-1>",self._on_release)

        # ── right: sidebar ────────────────────────────────────────────────────
        side = tk.Frame(main, bg=BG_CARD, width=230,
                        highlightbackground=ACCENT2, highlightthickness=1)
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)

        tk.Label(side, text="📦  Boxes in this image",
                 font=("Segoe UI", 10, "bold"), fg=ACCENT, bg=BG_CARD).pack(pady=(10, 4))

        list_f = tk.Frame(side, bg=BG_CARD)
        list_f.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        sb = tk.Scrollbar(list_f)
        sb.pack(side="right", fill="y")
        self.box_listbox = tk.Listbox(list_f, font=("Consolas", 9),
                                      bg="#0d1120", fg=TEXT, bd=0, relief="flat",
                                      selectbackground=ACCENT2, selectforeground="#fff",
                                      activestyle="none", yscrollcommand=sb.set)
        self.box_listbox.pack(side="left", fill="both", expand=True)
        sb.config(command=self.box_listbox.yview)
        self.box_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Class changer
        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=6, pady=6)
        tk.Label(side, text="Change selected class:",
                 font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", padx=10)
        self.class_var = tk.StringVar(value=CLASS_NAMES[0])
        cls_menu = ttk.Combobox(side, textvariable=self.class_var,
                                values=CLASS_NAMES, state="readonly", font=("Segoe UI", 9))
        cls_menu.pack(fill="x", padx=10, pady=4)
        cls_menu.bind("<<ComboboxSelected>>", self._change_class)

        # Buttons
        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=6, pady=6)
        self._sidebar_btn(side, "🗑️  Delete Selected Box", self._delete_selected, ERROR)
        self._sidebar_btn(side, "💾  Save Labels (S)",     self._save,            SUCCESS)

        # Navigation
        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=6, pady=6)
        nav = tk.Frame(side, bg=BG_CARD)
        nav.pack(fill="x", padx=8, pady=4)
        self._nav_btn(nav, "◀", self._prev).pack(side="left", expand=True, fill="x", padx=2)
        self._nav_btn(nav, "▶", self._next).pack(side="right", expand=True, fill="x", padx=2)

        # File list
        ttk.Separator(side, orient="horizontal").pack(fill="x", padx=6, pady=6)
        tk.Label(side, text="Images:", font=("Segoe UI", 9), fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", padx=10)
        file_f = tk.Frame(side, bg=BG_CARD)
        file_f.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        fsb = tk.Scrollbar(file_f)
        fsb.pack(side="right", fill="y")
        self.file_listbox = tk.Listbox(file_f, font=("Consolas", 8),
                                       bg="#0d1120", fg=TEXT_DIM, bd=0, relief="flat",
                                       selectbackground=ACCENT2, selectforeground="#fff",
                                       activestyle="none", yscrollcommand=fsb.set)
        self.file_listbox.pack(side="left", fill="both", expand=True)
        fsb.config(command=self.file_listbox.yview)
        for f in self.images:
            self.file_listbox.insert(tk.END, f"  {f}")
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        # Status bar
        self.status_var = tk.StringVar()
        sb2 = tk.Frame(self, bg=BG_CARD, height=24)
        sb2.pack(fill="x", side="bottom")
        tk.Label(sb2, textvariable=self.status_var, font=("Consolas", 9),
                 fg=TEXT_DIM, bg=BG_CARD, anchor="w", padx=12).pack(fill="x")

        # Key bindings
        self.bind("<Left>",       lambda e: self._prev())
        self.bind("<Right>",      lambda e: self._next())
        self.bind("<a>",          lambda e: self._prev())
        self.bind("<d>",          lambda e: self._next())
        self.bind("<s>",          lambda e: self._save())
        self.bind("<Control-s>",  lambda e: self._save())
        self.bind("<Delete>",     lambda e: self._delete_selected())
        self.bind("<BackSpace>",  lambda e: self._delete_selected())
        self.bind("<q>",          lambda e: self._quit())

    def _sidebar_btn(self, parent, text, cmd, col=ACCENT):
        b = tk.Button(parent, text=text, command=cmd, font=("Segoe UI", 9, "bold"),
                      fg=col, bg="#1e2540", activebackground="#2a3260",
                      activeforeground=col, relief="flat", bd=0, pady=7, cursor="hand2")
        b.pack(fill="x", padx=10, pady=2)
        return b

    def _nav_btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd, font=("Segoe UI", 12, "bold"),
                         fg=ACCENT, bg="#1e2540", activebackground="#2a3260",
                         activeforeground=ACCENT, relief="flat", bd=0,
                         padx=12, pady=6, cursor="hand2")

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_image(self, idx):
        self.idx     = idx
        fname        = self.images[idx]
        img_path     = os.path.join(IMG_DIR, fname)
        lbl_path     = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            self._set_status(f"Cannot read {fname}", ERROR)
            return

        self.orig_h, self.orig_w = img.shape[:2]
        self.current_cv = img
        self.lbl_path   = lbl_path
        self.boxes      = _read_label(lbl_path)
        self.selected   = -1
        self.dirty      = False

        self._update_file_list_highlight()
        self._refresh_display()
        self._refresh_title()
        self._set_status(f"{fname}  —  {len(self.boxes)} box(es)  |  "
                         "Click+drag to draw a new box")

    def _refresh_display(self):
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(),  400)
        ch = max(self.canvas.winfo_height(), 300)

        scale = min(cw / self.orig_w, ch / self.orig_h)
        dw    = int(self.orig_w * scale)
        dh    = int(self.orig_h * scale)
        self.disp_scale = scale
        self.disp_x0    = (cw - dw) // 2
        self.disp_y0    = (ch - dh) // 2

        # Draw image
        rgb  = cv2.cvtColor(self.current_cv, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb).resize((dw, dh), Image.LANCZOS)
        draw = ImageDraw.Draw(pil)

        # Draw boxes
        for i, box in enumerate(self.boxes):
            _, cx, cy, bw, bh = box
            x1 = int((cx - bw/2) * dw)
            y1 = int((cy - bh/2) * dh)
            x2 = int((cx + bw/2) * dw)
            y2 = int((cy + bh/2) * dh)

            col  = BOX_COLORS[box[0] % len(BOX_COLORS)]
            thick = 4 if i == self.selected else 2

            for t in range(thick):
                draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=col)

            name  = CLASS_NAMES[box[0]] if box[0] < len(CLASS_NAMES) else f"cls{box[0]}"
            label = f" {name} "
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("arialbd.ttf", 14)
            except Exception:
                font = None
            if font:
                tb = draw.textbbox((x1, max(y1-20, 0)), label, font=font)
            else:
                tb = (x1, max(y1-16, 0), x1 + len(label)*7, max(y1-16, 0)+14)
            draw.rectangle(tb, fill=col)
            draw.text((tb[0], tb[1]), label, fill=(10, 12, 26), font=font)

            # Selection highlight
            if i == self.selected:
                draw.rectangle([x1-6, y1-6, x2+6, y2+6],
                               outline=(255, 255, 255))

        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.disp_x0, self.disp_y0,
                                 anchor="nw", image=self._photo)

        # Refresh listbox
        self.box_listbox.delete(0, tk.END)
        for i, b in enumerate(self.boxes):
            name  = CLASS_NAMES[b[0]] if b[0] < len(CLASS_NAMES) else f"cls{b[0]}"
            col   = _hex(BOX_COLORS[b[0] % len(BOX_COLORS)])
            entry = f"  #{i+1}  {name}"
            self.box_listbox.insert(tk.END, entry)
            self.box_listbox.itemconfig(i, fg=col)
        if self.selected >= 0:
            self.box_listbox.selection_clear(0, tk.END)
            self.box_listbox.selection_set(self.selected)
            self.box_listbox.see(self.selected)

    # ── canvas interaction ────────────────────────────────────────────────────

    def _canvas_to_frac(self, cx, cy):
        """Convert canvas coordinates → fraction of original image (0..1)."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        dw = int(self.orig_w * self.disp_scale)
        dh = int(self.orig_h * self.disp_scale)
        fx = (cx - self.disp_x0) / dw
        fy = (cy - self.disp_y0) / dh
        return fx, fy

    def _hit_test(self, fx, fy):
        """Return index of box hit by fractional coords, or -1."""
        for i in range(len(self.boxes) - 1, -1, -1):   # top-most first
            _, cx, cy, bw, bh = self.boxes[i]
            if (cx - bw/2) <= fx <= (cx + bw/2) and \
               (cy - bh/2) <= fy <= (cy + bh/2):
                return i
        return -1

    def _on_click(self, event):
        fx, fy = self._canvas_to_frac(event.x, event.y)
        hit = self._hit_test(fx, fy)
        if hit >= 0:
            self.selected    = hit
            self._drawing    = False
            self._draw_start = None
            cls_id = self.boxes[hit][0]
            if cls_id < len(CLASS_NAMES):
                self.class_var.set(CLASS_NAMES[cls_id])
        else:
            # Start drawing a new box
            self.selected    = -1
            self._drawing    = True
            self._draw_start = (fx, fy)
        self._refresh_display()

    def _on_drag(self, event):
        if not self._drawing or self._draw_start is None:
            return
        fx, fy = self._canvas_to_frac(event.x, event.y)
        # Draw a preview rectangle on the canvas
        x0, y0 = self._draw_start
        dw = int(self.orig_w * self.disp_scale)
        dh = int(self.orig_h * self.disp_scale)
        px0 = self.disp_x0 + int(x0 * dw)
        py0 = self.disp_y0 + int(y0 * dh)
        px1 = self.disp_x0 + int(fx * dw)
        py1 = self.disp_y0 + int(fy * dh)
        self.canvas.delete("preview")
        self.canvas.create_rectangle(px0, py0, px1, py1,
                                     outline="#00e5ff", width=2, tags="preview")

    def _on_release(self, event):
        if not self._drawing or self._draw_start is None:
            return
        self._drawing = False
        fx, fy = self._canvas_to_frac(event.x, event.y)
        sx, sy = self._draw_start
        self._draw_start = None

        x1, x2 = min(sx, fx), max(sx, fx)
        y1, y2 = min(sy, fy), max(sy, fy)

        # Ignore tiny accidental drags
        if (x2 - x1) < 0.01 or (y2 - y1) < 0.01:
            return

        cls_id  = CLASS_NAMES.index(self.class_var.get()) \
                  if self.class_var.get() in CLASS_NAMES else 0
        cx      = (x1 + x2) / 2
        cy      = (y1 + y2) / 2
        bw      = x2 - x1
        bh      = y2 - y1
        self.boxes.append([cls_id, round(cx,6), round(cy,6),
                           round(bw,6), round(bh,6)])
        self.selected = len(self.boxes) - 1
        self.dirty    = True
        self._refresh_display()
        self._refresh_title()

    def _on_resize(self, event):
        self._refresh_display()

    # ── box operations ────────────────────────────────────────────────────────

    def _on_list_select(self, event):
        sel = self.box_listbox.curselection()
        if sel:
            self.selected = sel[0]
            cls_id = self.boxes[self.selected][0]
            if cls_id < len(CLASS_NAMES):
                self.class_var.set(CLASS_NAMES[cls_id])
            self._refresh_display()

    def _on_file_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            new_idx = sel[0]
            if new_idx != self.idx:
                self._navigate_away(new_idx)

    def _change_class(self, event=None):
        if self.selected < 0:
            return
        new_cls = CLASS_NAMES.index(self.class_var.get())
        self.boxes[self.selected][0] = new_cls
        self.dirty = True
        self._refresh_display()
        self._refresh_title()

    def _delete_selected(self):
        if self.selected < 0:
            self._set_status("No box selected. Click a box first.", WARNING)
            return
        del self.boxes[self.selected]
        self.selected = -1
        self.dirty    = True
        self._refresh_display()
        self._refresh_title()
        self._set_status(f"Box deleted. {len(self.boxes)} box(es) remaining.")

    def _save(self):
        _write_label(self.lbl_path, self.boxes)
        self.dirty = False
        self._refresh_title()
        self._set_status(f"✅  Saved {len(self.boxes)} box(es) → {self.lbl_path}")

    # ── navigation ────────────────────────────────────────────────────────────

    def _navigate_away(self, new_idx):
        if self.dirty:
            ans = messagebox.askyesnocancel(
                "Unsaved changes",
                f"Save changes to\n{self.images[self.idx]}\nbefore moving on?")
            if ans is None:   # Cancel
                return
            if ans:
                self._save()
        self._load_image(new_idx)

    def _prev(self):
        if self.idx > 0:
            self._navigate_away(self.idx - 1)

    def _next(self):
        if self.idx < len(self.images) - 1:
            self._navigate_away(self.idx + 1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _update_file_list_highlight(self):
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(self.idx)
        self.file_listbox.see(self.idx)

    def _refresh_title(self):
        fname = self.images[self.idx]
        self.counter_lbl.config(
            text=f"Image {self.idx+1} / {len(self.images)}  —  {fname}")
        if self.dirty:
            self.dirty_lbl.config(text="● unsaved")
        else:
            self.dirty_lbl.config(text="")

    def _set_status(self, msg, col=TEXT_DIM):
        self.status_var.set(f"  {msg}")

    def _quit(self):
        if self.dirty:
            ans = messagebox.askyesnocancel(
                "Unsaved changes",
                f"Save changes to {self.images[self.idx]} before quitting?")
            if ans is None:
                return
            if ans:
                self._save()
        self.destroy()


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)
    app = LabelReviewer()
    app.mainloop()
