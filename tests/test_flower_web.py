import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("onnxruntime")
pytest.importorskip("huggingface_hub")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402 (import after importorskip)

import flower_web  # noqa: E402 (import after importorskip)

# No `with TestClient(...) as client:` here on purpose: that would run the
# `startup` handler, which downloads the real model from the Hugging Face Hub.
client = TestClient(flower_web.app)


class DummyInput:
    name = "input"


class DummySession:
    """Stands in for an onnxruntime.InferenceSession."""

    def __init__(self, prediction):
        self._prediction = prediction
        self.last_feed = None

    def get_inputs(self):
        return [DummyInput()]

    def run(self, output_names, feed):
        self.last_feed = feed
        return [self._prediction]


def _jpeg_bytes(height=50, width=60):
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_index_returns_upload_form():
    response = client.get("/")
    assert response.status_code == 200
    assert "Flower classifier" in response.text
    assert 'action="/predict"' in response.text


def test_predict_returns_highest_probability_class(monkeypatch):
    prediction = np.array([[0.1, 0.05, 0.7, 0.1, 0.05]])
    session = DummySession(prediction)
    monkeypatch.setitem(flower_web.state, "session", session)
    monkeypatch.setitem(
        flower_web.state, "class_names",
        ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
    )

    predicted_class, probabilities = flower_web.predict(_jpeg_bytes())

    assert predicted_class == "roses"
    assert list(probabilities) == list(prediction[0])
    batch = session.last_feed["input"]
    assert batch.shape == (1, flower_web.HEIGHT, flower_web.WIDTH, 3)
    assert batch.dtype == np.float32


def test_image_to_data_uri_produces_jpeg_data_uri():
    data_uri = flower_web.image_to_data_uri(_jpeg_bytes())

    assert data_uri.startswith("data:image/jpeg;base64,")


def test_download_model_uses_configured_repo(monkeypatch):
    calls = {}

    def fake_download(repo_id, filename):
        calls.update(repo_id=repo_id, filename=filename)
        return "/tmp/flower_model.onnx"

    monkeypatch.setattr(flower_web, "hf_hub_download", fake_download)

    assert flower_web.download_model() == "/tmp/flower_model.onnx"
    assert calls == {"repo_id": "gdepaepe/dai6", "filename": "flower_model.onnx"}


def test_load_model_fills_state(monkeypatch):
    monkeypatch.setattr(flower_web, "download_model", lambda: "model.onnx")
    monkeypatch.setattr(flower_web.ort, "InferenceSession", lambda path: DummySession(None))
    monkeypatch.setitem(flower_web.state, "session", None)
    monkeypatch.setitem(flower_web.state, "class_names", None)

    flower_web.load_model()

    assert isinstance(flower_web.state["session"], DummySession)
    assert flower_web.state["class_names"] == ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
