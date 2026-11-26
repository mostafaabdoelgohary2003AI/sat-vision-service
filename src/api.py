"""FastAPI inference service for satellite image classification."""
import io, json, os
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from torchvision import models, transforms
from . import db
from .data import CLASSES

app = FastAPI(title="sat-vision-service", version="1.0.0")
_model = None
_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def model():
    global _model
    if _model is None:
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        sd = torch.load("results/model.pt", map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model"])
        m.eval()
        _model = m
    return _model

@app.on_event("startup")
def startup():
    db.run_migrations()

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.get("/metrics")
def metrics():
    return json.load(open("results/metrics.json"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = _tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model()(x), dim=1)[0]
    top = torch.topk(probs, 3)
    preds = [{"class": CLASSES[i], "probability": round(float(p), 4)}
             for p, i in zip(top.values, top.indices)]
    db.log_prediction(file.filename or "upload", preds[0]["class"], preds[0]["probability"])
    return {"filename": file.filename, "predictions": preds}

@app.get("/history")
def history(limit: int = 50):
    return {"predictions": db.history(limit)}

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return f.read()
