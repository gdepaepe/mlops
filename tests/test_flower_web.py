import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient

import flower_web

# No `with TestClient(...) as client:` here on purpose: that would run the
# `startup` handler, which tries to reach a real MLflow server.
client = TestClient(flower_web.app)


class DummyModel:
    def __init__(self, prediction):
        self._prediction = prediction

    def predict(self, batch):
        return self._prediction


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
    monkeypatch.setitem(flower_web.state, "model", DummyModel(prediction))
    monkeypatch.setitem(
        flower_web.state, "class_names",
        ["daisy", "dandelion", "roses", "sunflowers", "tulips"],
    )

    predicted_class, probabilities = flower_web.predict(_jpeg_bytes())

    assert predicted_class == "roses"
    assert list(probabilities) == list(prediction[0])


def test_image_to_data_uri_produces_jpeg_data_uri():
    data_uri = flower_web.image_to_data_uri(_jpeg_bytes())

    assert data_uri.startswith("data:image/jpeg;base64,")


def test_get_class_names_extracts_class_list(monkeypatch):
    monkeypatch.setattr(
        flower_web.mlflow.artifacts, "load_dict",
        lambda uri: {"class_names": ["daisy", "roses"]},
    )

    assert flower_web.get_class_names("run-123") == ["daisy", "roses"]


class FakeInfo:
    def __init__(self, run_id):
        self.run_id = run_id


class FakeRun:
    def __init__(self, run_id):
        self.info = FakeInfo(run_id)


class FakeExperiment:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id


class FakeClient:
    def __init__(self, experiment=None, runs=None):
        self._experiment = experiment
        self._runs = runs or []

    def get_experiment_by_name(self, name):
        return self._experiment

    def search_runs(self, experiment_ids, filter_string, order_by, max_results):
        return self._runs


def _patch_client(monkeypatch, fake_client):
    monkeypatch.setattr(flower_web.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(flower_web.mlflow.tracking, "MlflowClient", lambda: fake_client)


def test_get_latest_run_raises_when_experiment_missing(monkeypatch):
    _patch_client(monkeypatch, FakeClient(experiment=None))

    with pytest.raises(RuntimeError, match="not found"):
        flower_web.get_latest_run()


def test_get_latest_run_raises_when_no_finished_runs(monkeypatch):
    _patch_client(monkeypatch, FakeClient(experiment=FakeExperiment("exp1"), runs=[]))

    with pytest.raises(RuntimeError, match="No finished runs"):
        flower_web.get_latest_run()


def test_get_latest_run_returns_latest_run(monkeypatch):
    fake_run = FakeRun("run-abc")
    _patch_client(monkeypatch, FakeClient(experiment=FakeExperiment("exp1"), runs=[fake_run]))

    assert flower_web.get_latest_run() is fake_run
