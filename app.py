from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

model = YOLO("Microplastic_Yolov8_Model.pt")

PIXEL_TO_NM = 100
RISK_THRESHOLD = 15

@app.post("/detect")
async def detect_microplastic(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    results = model(img)
    boxes = results[0].boxes

    sizes_nm = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        w_nm = (x2 - x1) * PIXEL_TO_NM
        h_nm = (y2 - y1) * PIXEL_TO_NM
        size_nm = np.sqrt(w_nm * h_nm)
        sizes_nm.append(float(size_nm))

    min_count = sum(s <= min(sizes_nm)*1.1 for s in sizes_nm) if sizes_nm else 0
    max_count = sum(s >= max(sizes_nm)*0.9 for s in sizes_nm) if sizes_nm else 0
    avg_count = len(sizes_nm) - min_count - max_count

    risk_score = (min_count*3)+(avg_count*2)+(max_count*1)
    status = "UNSAFE" if risk_score >= RISK_THRESHOLD else "SAFE"

    return {
        "total_count": len(sizes_nm),
        "min_size": min(sizes_nm) if sizes_nm else 0,
        "avg_size": sum(sizes_nm)/len(sizes_nm) if sizes_nm else 0,
        "max_size": max(sizes_nm) if sizes_nm else 0,
        "min_count": min_count,
        "avg_count": avg_count,
        "max_count": max_count,
        "risk_score": risk_score,
        "status": status
    }
