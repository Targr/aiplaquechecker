import io
import os
import base64
import json
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

def annotate_image_yolo(img: np.ndarray, plates: List[Dict[str, Any]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=3)
        draw.text((x1, y1-10), f"Plate {plate['plate_id']}", fill=(255,255,0))
        for feat in plate.get("features", []):
            fx1, fy1, fx2, fy2 = feat["bbox"]
            draw.rectangle([fx1, fy1, fx2, fy2], outline=(0,255,0), width=2)
            draw.text((fx1, fy1-10), "Feature", fill=(0,255,255))
    return pil

def process_yolo(image_bytes: bytes) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, verbose=False)[0]

    plates = []
    features = []

    # Separate detections into plates vs features
    for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = yolo_model.names[int(cls_id)]
        det = {
            "bbox": [x1, y1, x2, y2],
            "confidence": float(conf),
            "center_x": int((x1+x2)/2),
            "center_y": int((y1+y2)/2),
            "width": x2-x1,
            "height": y2-y1,
        }
        if label.lower() == "plate":
            det["features"] = []
            det["plate_id"] = len(plates) + 1
            plates.append(det)
        else:
            features.append(det)

    # Assign features to their plates
    for feat in features:
        fx, fy = feat["center_x"], feat["center_y"]
        for plate in plates:
            x1, y1, x2, y2 = plate["bbox"]
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                plate["features"].append(feat)
                break  # assign to only one plate

    return plates, img_rgb

# --------- Core Processing ---------
def process_single_image(image_bytes: bytes, params: Dict[str, Any], reference_bytes: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    plates, img_rgb = process_yolo(image_bytes)
    annotated = annotate_image_yolo(img_rgb, plates)
    annotated_b64 = pil_to_base64(annotated)
    return {
        "plates": plates,
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
                total = len(result["plates"])
                rows.append({
                    "image_name": f.filename,
                    "total_features": total,
                    "plates": result["plates"]
                })
                img_data = base64.b64decode(result["annotated_image_base64"])
                zf.writestr(f.filename.replace(" ", "_"), img_data)
            except Exception as e:
                rows.append({
                    "image_name": f.filename,
                    "error": str(e),
                    "plates": []
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
    app.run(host="0.0.0.0", port=5000, debug=True)
