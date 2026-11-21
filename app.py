import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "model_config_2.json"

# Agronomic metadata mapped to your 4 target classes
CLASS_METADATA: Dict[str, Dict[str, str]] = {
    "drydown": {
        "label": "Drydown",
        "description": "Premature crop senescence, moisture stress, or rapid canopy drying.",
        "action": "Inspect irrigation uniformity, soil moisture retention, and drainage zones."
    },
    "double_plant": {
        "label": "Double Plant",
        "description": "Overlapping seed placement causing excessive plant population density.",
        "action": "Check planter meter calibration, singulator settings, and vacuum pressure."
    },
    "weed_cluster": {
        "label": "Weed Cluster",
        "description": "Localized weed patch competing with crops for nutrients and light.",
        "action": "Schedule targeted spot-spraying or mechanical cultivation in affected rows."
    },
    "planter_skip": {
        "label": "Planter Skip",
        "description": "Unplanted gap along crop rows due to seed drop interruption.",
        "action": "Inspect seed tubes, hoppers, and drive chains for blockages or mechanical slips."
    },
}


def resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    default_path = BASE_DIR / "model_2.keras"
    if default_path.exists():
        return default_path

    # Fallback: pick the first .keras file in the project root
    keras_files = sorted(BASE_DIR.glob("*.keras"))
    if keras_files:
        return keras_files[0]

    raise FileNotFoundError(
        "No .keras model file found. Place 'model_2.keras' in the project root or set MODEL_PATH."
    )


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()
CLASSES = CONFIG["classes"]
THRESHOLDS = CONFIG["thresholds"]
IMG_SIZE = int(CONFIG.get("img_size", 224))

MODEL_PATH = resolve_model_path()
# compile=False avoids deserializing the custom 'weighted_bce' training loss during inference
MODEL = keras.models.load_model(MODEL_PATH, compile=False)

app = FastAPI(title="Crop Anomaly Detection API", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)

    # MobileNetV2 expects inputs scaled to [-1, 1]
    # (If you trained with [0, 1] rescaling instead, replace the line below with: arr = arr / 255.0)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_file = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    return {
        "model_file": MODEL_PATH.name,
        "backbone": "MobileNetV2 (1.00)",
        "img_size": IMG_SIZE,
        "classes": [
            {
                "id": cls_name,
                "threshold": float(THRESHOLDS[cls_name]),
                **CLASS_METADATA.get(
                    cls_name,
                    {
                        "label": cls_name.replace("_", " ").title(),
                        "description": "",
                        "action": ""
                    },
                ),
            }
            for cls_name in CLASSES
        ],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    input_tensor = preprocess_image(image_bytes)

    start_time = time.perf_counter()
    raw_preds = MODEL.predict(input_tensor, verbose=0)[0]
    inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

    results = []
    detected_count = 0

    for idx, cls_name in enumerate(CLASSES):
        prob = float(raw_preds[idx])
        thresh = float(THRESHOLDS[cls_name])
        detected = prob >= thresh
        if detected:
            detected_count += 1

        meta = CLASS_METADATA.get(
            cls_name,
            {
                "label": cls_name.replace("_", " ").title(),
                "description": "",
                "action": ""
            },
        )

        results.append(
            {
                "id": cls_name,
                "label": meta["label"],
                "probability": round(prob, 4),
                "confidence_pct": round(prob * 100, 1),
                "threshold": thresh,
                "threshold_pct": round(thresh * 100, 1),
                "detected": detected,
                "margin": round(prob - thresh, 4),
                "description": meta["description"],
                "action": meta["action"],
            }
        )

    # Sort highest probability first
    results.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "filename": file.filename,
        "inference_ms": inference_ms,
        "detected_count": detected_count,
        "status": "anomaly_detected" if detected_count > 0 else "healthy",
        "predictions": results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
