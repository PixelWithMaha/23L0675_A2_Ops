from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLEAN_LIST = Path("clean_file_list.csv")
SPLIT_DIR = Path("splits")


def create_splits(clean_list=CLEAN_LIST, split_dir=SPLIT_DIR):
    df = pd.read_csv(clean_list)
    class_names = sorted(df["class"].unique())
    df["label"] = df["class"].map({name: idx for idx, name in enumerate(class_names)})

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"]
    )

    split_dir.mkdir(exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part[["filepath", "class", "label"]].to_csv(split_dir / f"{name}.csv", index=False)
    return class_names


def read_split(name, split_dir=SPLIT_DIR):
    df = pd.read_csv(split_dir / f"{name}.csv")
    return df["filepath"].tolist(), df["label"].astype("int32").tolist()


def parse_image(file_path, label, normalize=True, image_size=IMAGE_SIZE):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    if normalize:
        img = img / 255.0
    return img, label


def get_augmentation_layer(value_range=(0, 1)):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.15, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomContrast(0.1, value_range=value_range, seed=SEED),
    ], name="augmentation")


def build_pipeline(file_paths, labels, is_training=True, normalize=True, batch_size=BATCH_SIZE,
                   image_size=IMAGE_SIZE, augment=True):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if is_training:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths), seed=SEED, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda p, l: parse_image(p, l, normalize, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    if is_training and augment:
        augmenter = get_augmentation_layer((0, 1) if normalize else (0, 255))
        dataset = dataset.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_data_splits(normalize=True, image_size=IMAGE_SIZE, augment=True):
    if not (SPLIT_DIR / "train.csv").exists():
        create_splits()
    class_names = sorted(pd.read_csv(SPLIT_DIR / "train.csv")["class"].unique())

    train_ds = build_pipeline(*read_split("train"), is_training=True, normalize=normalize,
                              image_size=image_size, augment=augment)
    val_ds = build_pipeline(*read_split("val"), is_training=False, normalize=normalize,
                            image_size=image_size)
    test_ds = build_pipeline(*read_split("test"), is_training=False, normalize=normalize,
                             image_size=image_size)
    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    class_names = create_splits()
    print(f"Classes: {class_names}  (0 = {class_names[0]}, 1 = {class_names[1]})\n")
    for name in ["train", "val", "test"]:
        df = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        counts = df["class"].value_counts().to_dict()
        print(f"{name:5s}: {len(df):5d} images  {counts}")

    train_ds, _, _, _ = get_data_splits()
    images, labels = next(iter(train_ds))
    print(f"\nOne training batch: images {images.shape}, labels {labels.shape}")
    print(f"Pixel range: {float(tf.reduce_min(images)):.2f} to {float(tf.reduce_max(images)):.2f}")