import io
import os
import base64
from typing import List, Dict, Tuple, Any
import cv2

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO
import zipfile
from collections import Counter

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="/")

# --- Load YOLO model ---
MODEL_PATH = "/Users/juicejambouree/Downloads/plate_detection_sorry_petey/finetune_12plates_sorry_petey/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

# --- Class names from the YOLO model ---
CLASS_NAMES = yolo_model.names  # e.g., {0: 'feature', 1: 'feature_many_plates'}

# --------- Image Utils ---------
def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def annotate_image_yolo(img: np.ndarray, detections: List[Dict[str, Any]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, max(0, y1 - 10)), f"{det['label']} {det['confidence']:.2f}", fill=(255, 255, 0))
    return pil

def process_yolo(image_bytes: bytes, conf_threshold: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, verbose=False, conf=conf_threshold)[0]
    detections = []

    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    max_area = img_area * 0.15  # max 15% of image area

    # --- First pass: gather all areas ---
    raw_detections = []
    for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
        area = width * height
        raw_detections.append((x1, y1, x2, y2, width, height, area, float(conf), int(cls_id)))

    if not raw_detections:
        return [], img_rgb

    # --- Compute mean + stddev of areas ---
    areas = [d[6] for d in raw_detections]
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    area_cutoff = mean_area + 2 * std_area

    # --- Second pass: filter ---
    for (x1, y1, x2, y2, width, height, area, conf, cls_id) in raw_detections:
        if area > max_area:  # absolute max rule
            continue
        if area > area_cutoff:  # outlier rule
            continue
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "label": CLASS_NAMES[cls_id],
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
            "width": width,
            "height": height,
            "area": area,
        })

    return detections, img_rgb

# --------- Color counting ---------
def count_detections_by_color(img_rgb: np.ndarray, detections: List[Dict[str, Any]], color_map: Dict[str, Tuple[int,int,int]]) -> Dict[str, int]:
    """
    Count detections by user-specified colors.
    """
    counts = Counter({color: 0 for color in color_map.keys()})
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        mean_color = crop.mean(axis=(0,1))  # RGB mean
        # Find closest user-defined color
        closest_color = min(color_map.keys(), key=lambda c: np.linalg.norm(np.array(color_map[c]) - mean_color))
        counts[closest_color] += 1
    
    return dict(counts)

# --------- Core Processing ---------
def process_single_image(image_bytes: bytes, params: Dict[str, Any], reference_bytes: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    conf_threshold = float(params.get("confidence", 0.25))  # slider value from frontend
    detections, img_rgb = process_yolo(image_bytes, conf_threshold)
    annotated = annotate_image_yolo(img_rgb, detections)
    annotated_b64 = pil_to_base64(annotated)

    # Optional color counting
    color_counts = {}
    user_colors = params.get("colors")  # ["red", "green"]
    if user_colors:
        color_map = {
            "red": (255,0,0),
            "green": (0,255,0),
            "blue": (0,0,255),
            "yellow": (255,255,0),
            "orange": (255,165,0)
        }
        selected_map = {c: color_map[c] for c in user_colors if c in color_map}
        color_counts = count_detections_by_color(img_rgb, detections, selected_map)

    return {
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "color_counts": color_counts
    }

# --------- Routes ---------
@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/process")
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_file = request.files["image"]
    conf_threshold = request.form.get("confidence", 0.25)  # slider value

    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    # Parse colors if provided
    color_param = request.form.get("colors")
    colors = []
    if color_param:
        try:
            colors = list(eval(color_param)) if isinstance(color_param, str) else color_param
        except:
            colors = []

    out = process_single_image(image_file.read(), {"confidence": conf_threshold, "colors": colors}, references)
    return jsonify(out)

@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    conf_threshold = request.form.get("confidence", 0.25)
    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    # Parse colors if provided
    color_param = request.form.get("colors")
    colors = []
    if color_param:
        try:
            colors = list(eval(color_param)) if isinstance(color_param, str) else color_param
        except:
            colors = []

    rows = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for f in files:
            try:
                result = process_single_image(f.read(), {"confidence": conf_threshold, "colors": colors}, references)
                total = len(result["detections"])
                rows.append({
                    "image_name": f.filename,
                    "total_features": total,
                    "detections": result["detections"],
                    "color_counts": result["color_counts"]
                })
                img_data = base64.b64decode(result["annotated_image_base64"])
                zf.writestr(f.filename.replace(" ", "_"), img_data)
            except Exception as e:
                rows.append({
                    "image_name": f.filename,
                    "error": str(e),
                    "detections": [],
                    "color_counts": {}
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
