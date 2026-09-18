import sys
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.tensorflow

from flower_data import get_augmented_data_dir, load_datasets

# MLflow writes emoji (e.g. the run-URL line) to stdout; the default Windows console
# encoding (cp1252) can't handle those, which crashes the script after a successful run.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def build_model(cfg):
    """Build and compile the transfer-learning model. Backbone + optimizer come from config (yaml)."""
    imported_model = instantiate(cfg.model.backbone)
    for layer in imported_model.layers:
        layer.trainable = False

    imported_model.summary()

    dnn_model = keras.Sequential()
    dnn_model.add(imported_model)
    dnn_model.add(Dense(cfg.model.dense_units, activation='relu'))
    dnn_model.add(Dense(cfg.model.num_classes, activation='softmax'))

    optimizer = instantiate(cfg.model.optimizer)
    dnn_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return dnn_model


def train_model(dnn_model, train_set, validation_set, epochs):
    """Train the model and return the training history."""
    history = dnn_model.fit(
        train_set,
        validation_data=validation_set,
        epochs=epochs
    )
    return history


def log_history_to_mlflow(history):
    """Log every epoch's metrics (loss/accuracy, train + validation) to MLflow.

    Logged explicitly here rather than via mlflow.tensorflow.autolog(): autolog logs
    metrics through an internal async queue that, on this setup, turned out to
    sometimes drop metrics silently (e.g. when disabling autolog again shortly after
    training, to silence its warning about the later predict() call). Logging metrics
    ourselves from the returned history is simple and doesn't depend on that queue.
    """
    num_epochs = len(next(iter(history.history.values())))
    for epoch in range(num_epochs):
        mlflow.log_metrics(
            {name: values[epoch] for name, values in history.history.items()},
            step=epoch,
        )


def predict_image(dnn_model, image_path, image_cat, height, width):
    """Run inference on a single image and print the predicted category."""
    image = cv2.imread(str(image_path))
    image_resized = cv2.resize(image, (width, height))
    image = np.expand_dims(image_resized, axis=0)
    print(image.shape)

    model_pred = dnn_model.predict(image)
    print(model_pred)
    predicted_class = image_cat[np.argmax(model_pred)]
    print("The predicted category is", predicted_class)
    return predicted_class


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    height, width = cfg.data.height, cfg.data.width

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log the hyperparameters that matter for this run, plus the full resolved
        # config (including the imported model) as an artifact for reproducibility.
        mlflow.log_params({
            "height": height,
            "width": width,
            "batch_size": cfg.data.batch_size,
            "copies_per_image": cfg.data.copies_per_image,
            "dense_units": cfg.model.dense_units,
            "num_classes": cfg.model.num_classes,
            "epochs": cfg.model.epochs,
            "learning_rate": cfg.model.optimizer.learning_rate,
            "backbone": cfg.model.backbone._target_,
        })
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config.yaml")

        augmented_data = get_augmented_data_dir(cfg.data.dataset_dirname, cfg.data.augmented_dirname)
        train_set, validation_set, image_cat = load_datasets(augmented_data, height, width, cfg.data.batch_size)

        dnn_model = build_model(cfg)
        history = train_model(dnn_model, train_set, validation_set, cfg.model.epochs)
        log_history_to_mlflow(history)
        mlflow.tensorflow.log_model(dnn_model, name="model")
        # Log the class names next to the model so anything serving the model (e.g.
        # flower_web.py) can get them from MLflow instead of needing the training
        # dataset on disk.
        mlflow.log_dict({"class_names": list(image_cat)}, "class_names.json")

        sample_image = list((augmented_data / 'sunflowers').glob('*'))[1]
        predicted_class = predict_image(dnn_model, sample_image, image_cat, height, width)
        mlflow.log_param("sample_prediction", predicted_class)


if __name__ == "__main__":
    main()
