# file: app.py
import io
import os
import json
import base64
from typing import List, Dict, Tuple, Any, Optional
import threading
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO
import zipfile

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="/")

# --- Load YOLO model ---
MODEL_PATH = "/Users/juicejambouree/Downloads/plate_detection/finetune_12plates/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

# --- Class names from the YOLO model ---
CLASS_NAMES = yolo_model.names  # e.g., {0: 'feature', 1: 'feature_many_plates'}

# --- Paths for feedback/learning ---
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
FEEDBACK_PATH = DATA_DIR / "feedback_store.jsonl"
LEARNED_PATH = DATA_DIR / "learned_params.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------- Online Learner (lightweight) ---------
class OnlineLearner:
    """Maintains per-class stats from user feedback to filter/rescore detections.

    Why: Avoid costly on-the-fly model training; use feature stats (area ratio, aspect ratio, conf)
    to suppress frequent false positives and adapt thresholds per class.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.params: Dict[str, Any] = {
            "per_class": {},  # class -> dict with tp/fp stats and dynamic thresholds
            "global": {
                "default_max_area_ratio": 0.15,  # initial absolute cap
                "min_support": 10,               # required examples to trust class stats
            },
        }
        self._load()

    # --- Persistence ---
    def _load(self):
        if LEARNED_PATH.exists():
            try:
                with open(LEARNED_PATH, "r") as f:
                    self.params = json.load(f)
            except Exception:
                pass

    def _save(self):
        tmp = json.dumps(self.params, indent=2)
        with open(LEARNED_PATH, "w") as f:
            f.write(tmp)

    # --- Update stats from labeled example ---
    def update_with_example(self, label: str, bbox: List[int], img_w: int, img_h: int, conf: Optional[float], is_true_positive: bool):
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area_ratio = (w * h) / float(img_w * img_h)
        aspect_ratio = w / float(h)
        feat = {
            "area_ratio": float(area_ratio),
            "aspect_ratio": float(aspect_ratio),
            "conf": float(conf) if conf is not None else None,
        }
        with self.lock:
            pc = self.params["per_class"].setdefault(label, {
                "tp": {"area_ratio": [], "aspect_ratio": [], "conf": []},
                "fp": {"area_ratio": [], "aspect_ratio": [], "conf": []},
                "conf_floor": None,           # learned min conf
                "max_area_ratio": None,       # learned absolute cap
            })
            bucket = "tp" if is_true_positive else "fp"
            for k, v in feat.items():
                if v is not None:
                    pc[bucket][k].append(v)
            self._recompute_class_params(label)
            self._save()

    def _recompute_class_params(self, label: str):
        pc = self.params["per_class"][label]
        min_support = self.params["global"]["min_support"]
        # conf_floor: 10th percentile of TP conf if TPs >= min_support
        tp_conf = np.array(pc["tp"]["conf"], dtype=float) if pc["tp"]["conf"] else np.array([])
        if tp_conf.size >= max(5, min_support // 2):
            pc["conf_floor"] = float(np.percentile(tp_conf, 10))
        # max_area_ratio: mean + 2*std of TP area_ratio if supported
        tp_ar = np.array(pc["tp"]["area_ratio"], dtype=float) if pc["tp"]["area_ratio"] else np.array([])
        if tp_ar.size >= min_support:
            mean_ar = tp_ar.mean()
            std_ar = tp_ar.std()
            pc["max_area_ratio"] = float(min(0.33, mean_ar + 2 * std_ar))
        # If FP cluster small area dominates, optionally lower cap
        fp_ar = np.array(pc["fp"]["area_ratio"], dtype=float) if pc["fp"]["area_ratio"] else np.array([])
        if fp_ar.size >= min_support and tp_ar.size >= min_support:
            # If many FP areas are larger than TPs, tighten cap
            if np.median(fp_ar) > np.median(tp_ar):
                pc["max_area_ratio"] = float(min(pc.get("max_area_ratio", 0.33), np.median(tp_ar) + 1.5 * tp_ar.std()))

    # --- Scoring/Filter ---
    def filter_keep(self, label: str, area_ratio: float, aspect_ratio: float, conf: float) -> bool:
        with self.lock:
            pc = self.params["per_class"].get(label)
            default_max = self.params["global"]["default_max_area_ratio"]
            max_ratio = pc.get("max_area_ratio") if pc else None
            conf_floor = pc.get("conf_floor") if pc else None
        # absolute cap
        if area_ratio > (max_ratio if max_ratio is not None else default_max):
            return False
        # conf floor
        if conf_floor is not None and conf < conf_floor:
            return False
        return True

    def export_params(self) -> Dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.params))

learner = OnlineLearner()

# --------- Image Utils ---------
def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def annotate_image_yolo(img: np.ndarray, detections: List[Dict[str, Any]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        # why: expose IDs so frontends can map edits back to detections
        draw.text((x1, max(0, y1 - 12)), f"#{i} {det['label']} {det['confidence']:.2f}", fill=(255, 255, 0))
    return pil


def process_yolo(image_bytes: bytes, conf_threshold: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, verbose=False, conf=conf_threshold)[0]
    detections = []

    H, W = img_rgb.shape[:2]
    img_area = float(H * W)

    # --- First pass: gather all areas ---
    raw_detections = []
    for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
        area = width * height
        label = CLASS_NAMES[int(cls_id)]
        raw_detections.append({
            "bbox": [x1, y1, x2, y2],
            "width": width,
            "height": height,
            "area": area,
            "label": label,
            "confidence": float(conf),
        })

    if not raw_detections:
        return [], img_rgb

    # --- Compute mean + stddev of areas (image-specific outlier rule) ---
    areas = [d["area"] for d in raw_detections]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    area_cutoff = mean_area + 2 * std_area

    # --- Second pass: filter (image outliers + learned per-class constraints) ---
    for det in raw_detections:
        x1, y1, x2, y2 = det["bbox"]
        width, height, area = det["width"], det["height"], det["area"]
        area_ratio = area / img_area
        aspect_ratio = width / float(max(1, height))
        label = det["label"]
        conf = det["confidence"]

        # image-specific outlier rule
        if area > area_cutoff:
            continue
        # learned per-class rules
        if not learner.filter_keep(label, area_ratio, aspect_ratio, conf):
            continue
        det_out = {
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "label": label,
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
            "width": width,
            "height": height,
            "area": area,
        }
        detections.append(det_out)

    return detections, img_rgb


# --------- Core Processing ---------
def process_single_image(image_bytes: bytes, params: Dict[str, Any], reference_bytes: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    conf_threshold = float(params.get("confidence", 0.25))
    detections, img_rgb = process_yolo(image_bytes, conf_threshold)
    annotated = annotate_image_yolo(img_rgb, detections)
    annotated_b64 = pil_to_base64(annotated)
    return {
        "detections": detections,
        "annotated_image_base64": annotated_b64,
    }


# --------- Feedback I/O Helpers ---------
def _append_feedback(record: Dict[str, Any]):
    line = json.dumps(record, separators=(",", ":"))
    with open(FEEDBACK_PATH, "a") as f:
        f.write(line + "\n")


def _ingest_feedback_and_update_learner(record: Dict[str, Any]):
    # record: {
    #   image_name, image_w, image_h,
    #   accepted: [{bbox,label,confidence}],
    #   rejected: [{bbox,label,confidence}],
    #   added:    [{bbox,label}]  # no conf
    # }
    w = int(record.get("image_w"))
    h = int(record.get("image_h"))
    for det in record.get("accepted", []):
        learner.update_with_example(det["label"], det["bbox"], w, h, det.get("confidence"), True)
    for det in record.get("rejected", []):
        learner.update_with_example(det["label"], det["bbox"], w, h, det.get("confidence"), False)
    # Added boxes treated as TPs with neutral confidence
    for det in record.get("added", []):
        learner.update_with_example(det["label"], det["bbox"], w, h, None, True)


# --------- Routes ---------
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/process")
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_file = request.files["image"]
    conf_threshold = request.form.get("confidence", 0.25)

    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    out = process_single_image(image_file.read(), {"confidence": conf_threshold}, references)
    return jsonify(out)


@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    conf_threshold = request.form.get("confidence", 0.25)
    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    rows = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for f in files:
            try:
                result = process_single_image(f.read(), {"confidence": conf_threshold}, references)
                total = len(result["detections"])
                rows.append({
                    "image_name": f.filename,
                    "total_features": total,
                    "detections": result["detections"]
                })
                img_data = base64.b64decode(result["annotated_image_base64"])
                zf.writestr(f.filename.replace(" ", "_"), img_data)
            except Exception as e:
                rows.append({
                    "image_name": f.filename,
                    "error": str(e),
                    "detections": []
                })

    zip_buf.seek(0)
    csv_df = pd.DataFrame(rows)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
    zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

    return jsonify({
        "results": rows,
        "csv_base64": csv_b64,
        "zip_base64": zip_b64
    })


# --- New: submit edits/feedback that update the online learner ---
@app.post("/api/feedback")
def api_feedback():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    required = ["image_name", "image_w", "image_h", "accepted", "rejected", "added"]
    if any(k not in payload for k in required):
        return jsonify({"error": f"Missing fields, required: {required}"}), 400

    # persist raw feedback for auditability
    try:
        _append_feedback(payload)
    except Exception as e:
        return jsonify({"error": f"Failed to append feedback: {e}"}), 500

    # update learner now
    try:
        _ingest_feedback_and_update_learner(payload)
    except Exception as e:
        return jsonify({"error": f"Failed to update learner: {e}"}), 500

    return jsonify({"status": "ok", "learned_params": learner.export_params()})


# --- New: Inspect learned parameters ---
@app.get("/api/learned")
def api_learned():
    return jsonify(learner.export_params())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
