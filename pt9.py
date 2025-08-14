import io
import os
import base64
import json
import zipfile
from typing import List, Dict, Tuple, Any
import cv2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="/")

# --- Load YOLO model ---
MODEL_PATH = "/Users/juicejambouree/Downloads/plate_detection/finetune_12plates/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)

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
        label = det["label"]
        conf = det["confidence"]
        color = (0, 255, 0) if label == "plate" else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1-10), f"{label} {conf:.2f}", fill=color)
    return pil

# --------- YOLO Processing ---------
def process_yolo(image_bytes: bytes) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, verbose=False)[0]

    detections = []
    for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        cls_name = yolo_model.names[int(cls_id)]  # Get actual class name from YOLO
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "label": cls_name,
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
            "width": x2 - x1,
            "height": y2 - y1,
        })
    return detections, img_rgb

# --------- Core Processing ---------
def process_single_image(image_bytes: bytes, params: Dict[str, Any], reference_bytes: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    detections, img_rgb = process_yolo(image_bytes)
    annotated = annotate_image_yolo(img_rgb, detections)
    annotated_b64 = pil_to_base64(annotated)

    # Separate plates and features for frontend clarity
    plates = [d for d in detections if d["label"].lower() == "plate"]
    features = [d for d in detections if d["label"].lower() != "plate"]

    return {
        "plates": plates,
        "features": features,
        "annotated_image_base64": annotated_b64,
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
    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    out = process_single_image(image_file.read(), {}, references)
    return jsonify(out)

@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    rows = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for f in files:
            try:
                result = process_single_image(f.read(), {}, references)
                total_features = len(result["features"])
                rows.append({
                    "image_name": f.filename,
                    "total_features": total_features,
                    "plates": result["plates"],
                    "features": result["features"]
                })
                img_data = base64.b64decode(result["annotated_image_base64"])
                zf.writestr(f.filename.replace(" ", "_"), img_data)
            except Exception as e:
                rows.append({
                    "image_name": f.filename,
                    "error": str(e),
                    "plates": [],
                    "features": []
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
