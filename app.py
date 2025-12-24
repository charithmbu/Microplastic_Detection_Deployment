from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from ultralytics import YOLO

# ---------------- APP INIT ----------------
app = FastAPI(
    title="Microplastic Detection API",
    description="YOLOv8-based Microplastic Detection System",
    version="1.0.0"
)

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Microplastic_Yolov8_Model.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- CONFIG ----------------
PIXEL_TO_NM = 100        # 1 pixel = 100 nm (adjust after calibration)
RISK_THRESHOLD = 15

# ---------------- ROOT ROUTE ----------------
@app.get("/")
def root():
    return {
        "status": "Microplastic Detection API is running",
        "model": "YOLOv8",
        "endpoints": ["/detect", "/docs"]
    }

# ---------------- DETECTION ROUTE ----------------
@app.post("/detect")
async def detect_microplastics(file: UploadFile = File(...)):
    try:
        # ---------- READ IMAGE ----------
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image file"}
            )

        # ---------- YOLO INFERENCE ----------
        results = model(img)
        boxes = results[0].boxes
        total_count = len(boxes)

        sizes_nm = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]

            width_px = float(x2 - x1)
            height_px = float(y2 - y1)

            width_nm = width_px * PIXEL_TO_NM
            height_nm = height_px * PIXEL_TO_NM

            size_nm = (width_nm * height_nm) ** 0.5
            sizes_nm.append(size_nm)

        # ---------- STATISTICS ----------
        if sizes_nm:
            min_size = min(sizes_nm)
            max_size = max(sizes_nm)
            avg_size = sum(sizes_nm) / len(sizes_nm)

            min_thresh = min_size * 1.10
            max_thresh = max_size * 0.90

            min_count = sum(s <= min_thresh for s in sizes_nm)
            max_count = sum(s >= max_thresh for s in sizes_nm)
            avg_count = total_count - min_count - max_count

            risk_score = (min_count * 3) + (avg_count * 2) + (max_count * 1)
            status = "UNSAFE" if risk_score >= RISK_THRESHOLD else "SAFE"
        else:
            min_size = max_size = avg_size = 0
            min_count = avg_count = max_count = 0
            risk_score = 0
            status = "SAFE"

        # ---------- RESPONSE ----------
        return {
            "total_microplastics": total_count,
            "min_size_nm": round(min_size, 2),
            "avg_size_nm": round(avg_size, 2),
            "max_size_nm": round(max_size, 2),
            "size_category_counts": {
                "min_size": min_count,
                "average_size": avg_count,
                "max_size": max_count
            },
            "risk_score": risk_score,
            "status": status
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
