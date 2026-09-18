import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import hydra
from omegaconf import DictConfig


def download_flowers_data(dataset_dirname):
    """Download the flower photos dataset and return its local path."""
    flowers_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    extracted_dir = tf.keras.utils.get_file(dataset_dirname, origin=flowers_url, untar=True)
    # The archive's own top-level folder is also called "flower_photos", so the actual
    # class subfolders live one level deeper than what get_file() returns.
    flowers_data = pathlib.Path(extracted_dir) / dataset_dirname
    print(flowers_data)
    return flowers_data


def build_augmentation(height, width):
    """Data augmentation used to generate extra training images: random rotation + random crop."""
    return keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomCrop(height, width),
    ], name="data_augmentation")


def augment_and_save_dataset(flowers_data, height, width, load_height, load_width,
                              augmented_dirname, copies_per_image, output_dir=None):
    """Expand the flower photos dataset on disk.

    For every original image, save a (center-cropped) copy of the original plus
    `copies_per_image` randomly rotated/cropped copies into a new directory that
    mirrors the original class-subfolder structure. Returns the new directory.
    """
    output_dir = pathlib.Path(output_dir) if output_dir else flowers_data.parent / augmented_dirname
    augmentation = build_augmentation(height, width)
    center_crop = layers.CenterCrop(height, width)

    class_dirs = [d for d in flowers_data.iterdir() if d.is_dir()]
    for class_dir in class_dirs:
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.glob('*'):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_resized = cv2.resize(image, (load_width, load_height))
            batch = np.expand_dims(image_resized, axis=0)

            original_cropped = center_crop(batch).numpy()[0]
            original_cropped = np.clip(original_cropped, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_class_dir / image_path.name), original_cropped)

            for i in range(copies_per_image):
                augmented = augmentation(batch, training=True).numpy()[0]
                augmented = np.clip(augmented, 0, 255).astype(np.uint8)
                augmented_path = out_class_dir / f"{image_path.stem}_aug{i}{image_path.suffix}"
                cv2.imwrite(str(augmented_path), augmented)

    print(f"Expanded dataset saved to {output_dir}")
    return output_dir


def get_augmented_data_dir(dataset_dirname, augmented_dirname):
    """Path where augment_and_save_dataset() writes the augmented dataset by default.

    Does not create or download anything - the dataset must already have been
    built on disk, e.g. by running this module standalone (`python flower_data.py`).
    """
    keras_cache_dir = pathlib.Path(os.path.expanduser('~')) / '.keras' / 'datasets'
    return keras_cache_dir / dataset_dirname / augmented_dirname


def load_datasets(flowers_data, height, width, batch_size):
    """Build the training and validation datasets from a (possibly augmented) flower photos directory."""
    train_set, validation_set = tf.keras.preprocessing.image_dataset_from_directory(
        flowers_data,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)
    image_cat = train_set.class_names
    print(image_cat)

    return train_set, validation_set, image_cat


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    """Standalone entry point: download the flower photos and build the augmented dataset on disk."""
    height, width = cfg.data.height, cfg.data.width
    load_height = height + cfg.data.load_margin
    load_width = width + cfg.data.load_margin

    flowers_data = download_flowers_data(cfg.data.dataset_dirname)
    augment_and_save_dataset(
        flowers_data, height, width, load_height, load_width,
        cfg.data.augmented_dirname, cfg.data.copies_per_image)


if __name__ == "__main__":
    main()
