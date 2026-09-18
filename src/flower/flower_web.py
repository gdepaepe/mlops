"""FastAPI website: upload a flower photo, get the predicted flower type back.

Loads the trained model from the latest finished MLflow run of the
"flowers-classification" experiment (see flower_model.py, which logs it there).
Run with:  uvicorn flower_web:app --reload
"""
import base64
import io
import os

import cv2
import numpy as np
import mlflow
import mlflow.tensorflow
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf
from PIL import Image

CONFIG = OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
HEIGHT, WIDTH = CONFIG.data.height, CONFIG.data.width

# Allow overriding the tracking URI via env var: config.yaml's "http://127.0.0.1:5000"
# only works when this process runs on the same host as `mlflow ui`. When running in
# Docker, point this at the host instead, e.g. MLFLOW_TRACKING_URI=http://host.docker.internal:5000
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", CONFIG.mlflow.tracking_uri)

app = FastAPI(title="Flower classifier")

# Populated on startup: the loaded Keras model and its class names (in the same
# order the model's output layer uses, i.e. the sorted class subfolder names).
state = {"model": None, "class_names": None, "run_id": None}


def get_latest_run():
    """Find the most recent finished run of the flowers-classification experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(CONFIG.mlflow.experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{CONFIG.mlflow.experiment_name}' not found. "
            "Run `python flower_model.py` at least once first."
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished runs found in experiment '{CONFIG.mlflow.experiment_name}'. "
            "Run `python flower_model.py` at least once first."
        )
    return runs[0]


def get_class_names(run_id):
    """Class names in the same order the model's output layer uses, as logged by
    flower_model.py's training run (avoids needing the training dataset on disk)."""
    return mlflow.artifacts.load_dict(f"runs:/{run_id}/class_names.json")["class_names"]


@app.on_event("startup")
def load_model():
    run = get_latest_run()
    state["run_id"] = run.info.run_id
    state["model"] = mlflow.tensorflow.load_model(f"runs:/{run.info.run_id}/model")
    state["class_names"] = get_class_names(run.info.run_id)
    print(f"Loaded model from MLflow run {run.info.run_id}, classes: {state['class_names']}")


def predict(image_bytes: bytes):
    """Preprocess uploaded image bytes and return (predicted_class, probabilities)."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (WIDTH, HEIGHT))
    batch = np.expand_dims(image_resized, axis=0)

    probabilities = state["model"].predict(batch)[0]
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
