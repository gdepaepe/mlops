from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_PATH = Path(__file__).resolve().parent.parent / "src" / "flower" / "config.yaml"


def load_config():
    return OmegaConf.load(CONFIG_PATH)


def test_data_section_defaults():
    config = load_config()
    assert config.data.height == 180
    assert config.data.width == 180
    assert config.data.load_margin == 20
    assert config.data.batch_size == 32
    assert config.data.copies_per_image == 5


def test_model_optimizer_target_and_learning_rate():
    config = load_config()
    assert config.model.optimizer._target_ == "tensorflow.keras.optimizers.Adam"
    assert config.model.optimizer.learning_rate == 0.0005


def test_backbone_input_shape_interpolates_from_data_section():
    config = load_config()
    resolved = OmegaConf.to_container(config, resolve=True)

    assert resolved["model"]["backbone"]["input_shape"] == [180, 180, 3]
    assert resolved["model"]["backbone"]["classes"] == resolved["model"]["num_classes"] == 5


def test_instantiate_builds_the_configured_optimizer():
    tf = pytest.importorskip("tensorflow")
    from hydra.utils import instantiate

    config = load_config()
    optimizer = instantiate(config.model.optimizer)

    assert isinstance(optimizer, tf.keras.optimizers.Adam)
    assert float(optimizer.learning_rate.numpy()) == pytest.approx(0.0005)
