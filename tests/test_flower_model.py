import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow")
pytest.importorskip("mlflow")

import flower_model


class DummyModel:
    """Stand-in for a trained Keras model, so tests don't need to train or load one."""

    def __init__(self, prediction):
        self._prediction = prediction
        self.received_shape = None

    def predict(self, image):
        self.received_shape = image.shape
        return self._prediction


class FakeHistory:
    def __init__(self, history_dict):
        self.history = history_dict


def _write_dummy_image(path, height=100, width=120):
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_predict_image_returns_highest_probability_category(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _write_dummy_image(image_path)

    prediction = np.array([[0.1, 0.05, 0.7, 0.1, 0.05]])
    model = DummyModel(prediction)
    image_cat = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    predicted = flower_model.predict_image(model, image_path, image_cat, height=180, width=180)

    assert predicted == "roses"
    assert model.received_shape == (1, 180, 180, 3)


def test_predict_image_picks_the_argmax_class(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _write_dummy_image(image_path)

    prediction = np.array([[0.6, 0.1, 0.1, 0.1, 0.1]])
    model = DummyModel(prediction)
    image_cat = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    predicted = flower_model.predict_image(model, image_path, image_cat, height=180, width=180)

    assert predicted == "daisy"


def test_log_history_to_mlflow_logs_metrics_per_epoch(monkeypatch):
    logged = []
    monkeypatch.setattr(
        flower_model.mlflow, "log_metrics",
        lambda metrics, step: logged.append((step, metrics)),
    )

    history = FakeHistory({
        "loss": [0.9, 0.5],
        "accuracy": [0.4, 0.8],
        "val_loss": [1.0, 0.6],
        "val_accuracy": [0.3, 0.7],
    })

    flower_model.log_history_to_mlflow(history)

    assert logged == [
        (0, {"loss": 0.9, "accuracy": 0.4, "val_loss": 1.0, "val_accuracy": 0.3}),
        (1, {"loss": 0.5, "accuracy": 0.8, "val_loss": 0.6, "val_accuracy": 0.7}),
    ]
