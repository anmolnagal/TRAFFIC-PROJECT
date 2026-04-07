"""
app.py — TrafficVision AI  |  Flask + SocketIO backend
=======================================================
Serves traffic_dashboard.html and wires it to the real YOLO model.

Run from project root:
    python demo/app.py

Then open:  http://localhost:5000
"""

import base64
import io
import json
import os
import sys
import threading
import time
import queue
import webbrowser
from collections import Counter, deque

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model paths ───────────────────────────────────────────────────────────────
CUSTOM_MODEL   = os.path.join(ROOT, "models", "indian_vehicles_yolo.pt")
FALLBACK_MODEL = os.path.join(ROOT, "yolov8n.pt")

# COCO → friendly name for fallback mode
COCO_TO_INDIAN = {
    0: "pedestrian", 1: "bicycle", 2: "car",
    3: "two_wheeler", 5: "bus", 7: "truck",
}

BOX_COLORS = [
    (0, 229, 255), (124, 77, 255), (0, 230, 118),
    (255, 214, 0), (255, 64, 129), (255, 111, 0),
    (0, 200, 83),  (213, 0, 249), (29, 233, 182),
    (255, 196, 0),
]

# ── Persistence ──────────────────────────────────────────────────────────────
PERSISTENCE_FILE = os.path.join(ROOT, "data", "detection_log.json")

def _load_persisted():
    """Load saved detection log and totals from disk."""
    global total_detections, class_totals, detection_log
    try:
        if os.path.exists(PERSISTENCE_FILE):
            with open(PERSISTENCE_FILE, "r") as f:
                saved = json.load(f)
            total_detections = saved.get("total", 0)
            class_totals = Counter(saved.get("class_totals", {}))
            for entry in reversed(saved.get("log", [])):
                detection_log.appendleft(entry)
            print(f"[TrafficVision] Loaded {total_detections} persisted detections")
    except Exception as exc:
        print(f"[TrafficVision] Could not load persisted data: {exc}")

def _save_persisted():
    """Write current detection log and totals to disk."""
    try:
        os.makedirs(os.path.dirname(PERSISTENCE_FILE), exist_ok=True)
        with open(PERSISTENCE_FILE, "w") as f:
            json.dump({
                "total":       total_detections,
                "class_totals": dict(class_totals),
                "log":         list(detection_log),
            }, f, indent=2)
    except Exception as exc:
        print(f"[TrafficVision] Could not save data: {exc}")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=DEMO_DIR, static_url_path="")
app.config["SECRET_KEY"] = "trafficvision-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024)  # 10 MB frames

# ── Global state ──────────────────────────────────────────────────────────────
yolo_model      = None
model_is_custom = False
model_name      = "loading..."
model_loading   = True

webcam_active   = False
webcam_cap      = None
webcam_thread   = None

# Cumulative stats
total_detections   = 0
detection_log      = deque(maxlen=1000)   # last 1000 detections
class_totals       = Counter()

# Load any previously saved data immediately
_load_persisted()

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    global yolo_model, model_is_custom, model_name, model_loading
    try:
        from ultralytics import YOLO
        if os.path.exists(CUSTOM_MODEL):
            mdl    = YOLO(CUSTOM_MODEL)
            custom = True
            name   = "indian_vehicles_yolo.pt"
        else:
            mdl    = YOLO(FALLBACK_MODEL)
            custom = False
            name   = "yolov8n.pt (COCO fallback)"

        yolo_model      = mdl
        model_is_custom = custom
        model_name      = name
        model_loading   = False
        print(f"[TrafficVision] Model loaded: {name}")
        socketio.emit("model_ready", {"name": name, "custom": custom}, namespace="/")
    except Exception as exc:
        model_loading = False
        print(f"[TrafficVision] Model load error: {exc}")
        socketio.emit("model_error", {"error": str(exc)}, namespace="/")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_results(results_raw):
    bboxes, classes, confs = [], [], []
    for box in results_raw.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if model_is_custom:
            name = results_raw.names.get(cls_id, f"class_{cls_id}")
        else:
            if cls_id not in COCO_TO_INDIAN:
                continue
            name = COCO_TO_INDIAN[cls_id]

        bboxes.append([x1, y1, x2, y2])
        classes.append(name)
        confs.append(round(conf, 3))
    return bboxes, classes, confs


def draw_cv(frame, bboxes, classes, confs):
    """Draw bounding boxes directly on a cv2 BGR frame (fast path)."""
    for i, (bbox, cls, conf) in enumerate(zip(bboxes, classes, confs)):
        x1, y1, x2, y2 = bbox
        col = BOX_COLORS[i % len(BOX_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

        label = f"{cls} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        bg_y = max(y1 - 20, 0)
        cv2.rectangle(frame, (x1, bg_y), (x1 + tw + 6, bg_y + th + 6), col, -1)
        cv2.putText(frame, label, (x1 + 3, bg_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 12, 26), 1, cv2.LINE_AA)
    return frame


def frame_to_b64(frame, quality: int = 72) -> str:
    """Encode a BGR cv2 frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def record_detections(classes, confs, source="webcam", bboxes=None):
    global total_detections
    total_detections += len(classes)
    class_totals.update(classes)
    ts = time.strftime("%H:%M:%S")
    date_str = time.strftime("%Y-%m-%d")
    for i, (cls, conf) in enumerate(zip(classes, confs)):
        bbox = bboxes[i] if bboxes and i < len(bboxes) else []
        detection_log.appendleft({
            "ts":     ts,
            "date":   date_str,
            "cls":    cls,
            "conf":   conf,
            "source": source,
            "bbox":   bbox,
            "frame":  total_detections - len(classes) + i + 1,
        })
    # Persist to disk (async to not block)
    threading.Thread(target=_save_persisted, daemon=True).start()


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(DEMO_DIR, "traffic_dashboard.html")


@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "name":    model_name,
        "custom":  model_is_custom,
        "loading": model_loading,
    })


@app.route("/api/stats")
def api_stats():
    total = total_detections
    avg_conf = 0.0
    if detection_log:
        avg_conf = sum(d["conf"] for d in detection_log) / len(detection_log)

    # Hourly buckets for today (00-23)
    # { hour: { class: count } }
    today = time.strftime("%Y-%m-%d")
    hourly = {h: Counter() for h in range(24)}
    
    for d in detection_log:
        if d.get("date") == today:
            try:
                h = int(d["ts"].split(":")[0])
                hourly[h][d["cls"]] += 1
            except (ValueError, KeyError):
                continue
    
    # Format for frontend: { hour: { class: count } }
    formatted_hourly = {str(h).zfill(2): dict(counts) for h, counts in hourly.items()}

    return jsonify({
        "total":          total,
        "avg_conf":       round(avg_conf * 100, 1),
        "class_totals":   dict(class_totals),
        "recent":         list(detection_log)[:10],
        "hourly_counts":  formatted_hourly
    })


@app.route("/api/log")
def api_log():
    """Paginated full detection log."""
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    log_list = list(detection_log)
    total    = len(log_list)
    start    = (page - 1) * per_page
    end      = start + per_page
    return jsonify({
        "total":      total,
        "page":       page,
        "per_page":   per_page,
        "pages":      max(1, -(-total // per_page)),   # ceil division
        "entries":    log_list[start:end],
    })


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Accept an uploaded image file, run YOLO, return JSON results."""
    if model_loading:
        return jsonify({"error": "Model still loading, please wait"}), 503
    if yolo_model is None:
        return jsonify({"error": "Model not available"}), 503

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Read image from upload
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    conf_thresh = float(request.form.get("conf", 0.25))

    try:
        results_raw = yolo_model(img, conf=conf_thresh, verbose=False)[0]
        bboxes, classes, confs = parse_results(results_raw)

        # Draw and encode annotated image
        annotated = draw_cv(img.copy(), bboxes, classes, confs)
        annotated_b64 = frame_to_b64(annotated, quality=85)

        record_detections(classes, confs, source=file.filename or "upload", bboxes=bboxes)

        return jsonify({
            "count":      len(bboxes),
            "classes":    classes,
            "confs":      confs,
            "bboxes":     bboxes,
            "class_counts": dict(Counter(classes)),
            "avg_conf":   round(sum(confs) / len(confs) * 100, 1) if confs else 0,
            "image":      annotated_b64,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── WebSocket: webcam stream ──────────────────────────────────────────────────

def webcam_worker():
    """Background thread: captures frames, runs YOLO, emits to browser."""
    global webcam_active, webcam_cap

    detect_every = 4
    frame_counter = 0
    last_boxes, last_classes, last_confs = [], [], []
    t_last = time.time()

    while webcam_active:
        ok, frame = webcam_cap.read()
        if not ok:
            break

        frame_counter += 1

        # Run inference every N frames (CPU-friendly)
        if frame_counter % detect_every == 0:
            try:
                conf_thresh = 0.25
                res = yolo_model(frame, conf=conf_thresh, verbose=False, imgsz=416)[0]
                last_boxes, last_classes, last_confs = parse_results(res)
                record_detections(last_classes, last_confs, source="webcam")
            except Exception:
                pass

        # Annotate frame
        annotated = draw_cv(frame.copy(), last_boxes, last_classes, last_confs)
        b64 = frame_to_b64(annotated, quality=65)

        avg_conf = (sum(last_confs) / len(last_confs)) if last_confs else 0.0
        class_counts = dict(Counter(last_classes))

        socketio.emit("frame", {
            "image":        b64,
            "count":        len(last_boxes),
            "avg_conf":     round(avg_conf * 100, 1),
            "classes":      last_classes,
            "class_counts": class_counts,
        }, namespace="/")

        # ~20 FPS cap
        elapsed = time.time() - t_last
        sleep   = max(0, (1 / 20) - elapsed)
        time.sleep(sleep)
        t_last = time.time()

    # Cleanup
    if webcam_cap:
        webcam_cap.release()
    webcam_cap = None
    socketio.emit("webcam_stopped", {}, namespace="/")
    print("[TrafficVision] Webcam stopped")


@socketio.on("start_webcam")
def on_start_webcam(data=None):
    global webcam_active, webcam_cap, webcam_thread

    if webcam_active:
        emit("webcam_error", {"error": "Webcam already running"})
        return
    if yolo_model is None:
        emit("webcam_error", {"error": "Model not loaded yet"})
        return

    raw_source = (data or {}).get("index", 0)
    
    # Smart source detection: if it's all digits, it's a camera index.
    # Otherwise, it's likely a stream URL (RTSP, HTTP, etc.)
    if str(raw_source).isdigit():
        source = int(raw_source)
        label = f"Camera {source}"
    else:
        source = str(raw_source)
        label = f"Remote Stream"

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        emit("webcam_error", {"error": f"Cannot open source: {source}"})
        return

    # Optimize capture size for local webcams (may not apply to some RTSP streams)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    webcam_cap    = cap
    webcam_active = True
    webcam_thread = threading.Thread(target=webcam_worker, daemon=True)
    webcam_thread.start()

    emit("webcam_started", {"index": raw_source, "label": label})
    print(f"[TrafficVision] Source started: {label} ({source})")


@socketio.on("stop_webcam")
def on_stop_webcam(data=None):
    global webcam_active
    webcam_active = False
    emit("webcam_stopping", {})
    print("[TrafficVision] Webcam stop requested")


@socketio.on("connect")
def on_connect():
    print("[TrafficVision] Browser client connected")
    emit("server_info", {
        "model_name":   model_name,
        "model_custom": model_is_custom,
        "model_loading": model_loading,
    })


@socketio.on("disconnect")
def on_disconnect():
    print("[TrafficVision] Browser client disconnected")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(ROOT)
    print("=" * 60)
    print("  TrafficVision AI — Command Center")
    print("  Starting server at http://localhost:5000")
    print("=" * 60)

    # Load model in background thread
    threading.Thread(target=load_model, daemon=True).start()

    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=open_browser, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
