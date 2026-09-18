import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tensorflow")

import flower_data


def test_get_augmented_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(flower_data.os.path, "expanduser", lambda p: str(tmp_path))

    result = flower_data.get_augmented_data_dir("flower_photos", "flower_photos_augmented")

    expected = tmp_path / ".keras" / "datasets" / "flower_photos" / "flower_photos_augmented"
    assert result == expected


def test_build_augmentation_layers():
    augmentation = flower_data.build_augmentation(180, 180)

    layer_types = [type(layer).__name__ for layer in augmentation.layers]
    assert layer_types == ["RandomRotation", "RandomCrop"]


def _write_dummy_image(path, size=50):
    image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_augment_and_save_dataset_creates_expected_files(tmp_path):
    flowers_data = tmp_path / "flower_photos"
    class_dir = flowers_data / "roses"
    class_dir.mkdir(parents=True)
    _write_dummy_image(class_dir / "rose1.jpg")

    output_dir = tmp_path / "augmented"
    result = flower_data.augment_and_save_dataset(
        flowers_data,
        height=180, width=180, load_height=200, load_width=200,
        augmented_dirname="unused", copies_per_image=2, output_dir=output_dir,
    )

    assert result == output_dir

    out_class_dir = output_dir / "roses"
    saved_files = sorted(p.name for p in out_class_dir.glob("*"))
    assert saved_files == ["rose1.jpg", "rose1_aug0.jpg", "rose1_aug1.jpg"]

    for saved_path in out_class_dir.glob("*"):
        saved_image = cv2.imread(str(saved_path))
        assert saved_image.shape == (180, 180, 3)


def test_augment_and_save_dataset_skips_unreadable_files(tmp_path):
    flowers_data = tmp_path / "flower_photos"
    class_dir = flowers_data / "tulips"
    class_dir.mkdir(parents=True)
    (class_dir / "not_an_image.txt").write_text("this is not an image")

    output_dir = tmp_path / "augmented"
    flower_data.augment_and_save_dataset(
        flowers_data,
        height=180, width=180, load_height=200, load_width=200,
        augmented_dirname="unused", copies_per_image=1, output_dir=output_dir,
    )

    out_class_dir = output_dir / "tulips"
    assert list(out_class_dir.glob("*")) == []


def test_augment_and_save_dataset_default_output_dir(tmp_path):
    flowers_data = tmp_path / "flower_photos"
    class_dir = flowers_data / "roses"
    class_dir.mkdir(parents=True)
    _write_dummy_image(class_dir / "rose1.jpg")

    result = flower_data.augment_and_save_dataset(
        flowers_data,
        height=180, width=180, load_height=200, load_width=200,
        augmented_dirname="flower_photos_augmented", copies_per_image=0,
    )

    assert result == flowers_data.parent / "flower_photos_augmented"
