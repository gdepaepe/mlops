"""FastAPI website: upload a flower photo, get the predicted flower type back.

Downloads the trained ONNX model from the Hugging Face Hub (repo configured in
config.yaml, by default https://huggingface.co/gdepaepe/dai6) and runs it with
onnxruntime. Run with:  uvicorn flower_web:app --reload
"""
import base64
import io
import os

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image

CONFIG = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
HEIGHT, WIDTH = CONFIG.data.height, CONFIG.data.width

app = FastAPI(title="Flower classifier")

# Populated on startup: the ONNX inference session and its class names (in the same
# order the model's output layer uses, i.e. the sorted class subfolder names).
state = {"session": None, "class_names": None}


def download_model():
    """Download the ONNX model from the Hugging Face Hub (cached locally after the
    first download) and return the local file path."""
    return hf_hub_download(repo_id=CONFIG.huggingface.repo_id, filename=CONFIG.huggingface.filename)


@app.on_event("startup")
def load_model():
    model_path = download_model()
    state["session"] = ort.InferenceSession(model_path)
    state["class_names"] = list(CONFIG.huggingface.class_names)
    print(f"Loaded model {CONFIG.huggingface.repo_id}/{CONFIG.huggingface.filename}, "
          f"classes: {state['class_names']}")


def predict(image_bytes: bytes):
    """Preprocess uploaded image bytes and return (predicted_class, probabilities)."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # The model was trained on RGB images (image_dataset_from_directory); OpenCV decodes as BGR.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (WIDTH, HEIGHT))
    batch = np.expand_dims(image_resized, axis=0).astype(np.float32)

    session = state["session"]
    input_name = session.get_inputs()[0].name
    probabilities = session.run(None, {input_name: batch})[0][0]
    predicted_class = state["class_names"][int(np.argmax(probabilities))]
    return predicted_class, probabilities


def image_to_data_uri(image_bytes: bytes) -> str:
    """Re-encode the uploaded image as a small embeddable JPEG data URI for the preview."""
    thumb = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    thumb.thumbnail((300, 300))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Flower classifier</title>
  <style>
    body {{ font-family: sans-serif; max-width: 480px; margin: 3rem auto; text-align: center; }}
    img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; }}
    .result {{ font-size: 1.4rem; margin: 1rem 0; }}
    .probs {{ text-align: left; font-size: 0.9rem; color: #444; }}
  </style>
</head>
<body>
  <h1>🌸 Flower classifier</h1>
  <form action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required onchange="this.form.submit()">
  </form>
  {content}
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE_TEMPLATE.format(content="")


# Parsing the uploaded file below relies on the python-multipart package being installed.
@app.post("/predict", response_class=HTMLResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_class, probabilities = predict(image_bytes)

    probs_html = "".join(
        f"<div>{name}: {prob:.1%}</div>"
        for name, prob in sorted(zip(state["class_names"], probabilities), key=lambda x: -x[1])
    )
    content = f"""
      <img src="{image_to_data_uri(image_bytes)}" alt="uploaded flower">
      <div class="result">Predicted: <strong>{predicted_class}</strong></div>
      <div class="probs">{probs_html}</div>
    """
    return PAGE_TEMPLATE.format(content=content)
