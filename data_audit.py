import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

DATA_DIR = Path("./data")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
TF_DECODABLE_FORMATS = {"JPEG", "PNG", "BMP", "GIF"}


def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def audit_dataset(data_dir):
    records, corrupt_files, skipped_files = [], [], []
    print(f"Auditing dataset at: {data_dir.resolve()}\n")

    for filepath in sorted(data_dir.rglob("*")):
        if not filepath.is_file():
            continue
        if filepath.name.startswith(".") or filepath.suffix.lower() not in VALID_EXTENSIONS:
            skipped_files.append(str(filepath))
            continue

        class_name = filepath.relative_to(data_dir).parts[0]

        try:
            with Image.open(filepath) as img:
                img.verify()
            with Image.open(filepath) as img:
                img.load()
                width, height = img.size
                color_mode = img.mode
                img_format = img.format
        except Exception as err:
            corrupt_files.append({"filepath": str(filepath), "class": class_name, "error": str(err)})
            continue

        records.append({
            "filepath": str(filepath),
            "class": class_name,
            "file_size_kb": filepath.stat().st_size / 1024.0,
            "width": width,
            "height": height,
            "color_mode": color_mode,
            "format": img_format,
            "sha256": compute_sha256(filepath),
        })

    df = pd.DataFrame(records)

    df["is_duplicate"] = df.duplicated(subset="sha256", keep="first")
    classes_per_hash = df.groupby("sha256")["class"].nunique()
    cross_class_hashes = set(classes_per_hash[classes_per_hash > 1].index)
    df["cross_class_duplicate"] = df["sha256"].isin(cross_class_hashes)
    df["tf_decodable"] = df["format"].isin(TF_DECODABLE_FORMATS)

    clean = df[df["tf_decodable"] & ~df["cross_class_duplicate"]].drop_duplicates(subset="sha256")

    df.to_csv("audit_dataset_metadata.csv", index=False)
    pd.DataFrame(corrupt_files).to_csv("audit_corrupt_files.csv", index=False)
    df[df["is_duplicate"]].to_csv("audit_duplicates.csv", index=False)
    clean[["filepath", "class"]].to_csv("clean_file_list.csv", index=False)

    line = "=" * 65
    print(line)
    print("TASK 1.2: DATASET AUDIT SUMMARY REPORT")
    print(line)
    print(f"Readable images                 : {len(df)}")
    print(f"Corrupt / unreadable files      : {len(corrupt_files)}")
    print(f"Skipped (hidden / other ext.)   : {len(skipped_files)}")
    print(f"Exact duplicate copies          : {int(df['is_duplicate'].sum())}")
    print(f"Images in cross-class duplicates: {int(df['cross_class_duplicate'].sum())}")
    print(f"Not decodable by TensorFlow     : {int((~df['tf_decodable']).sum())}")
    print(f"Clean images kept for training  : {len(clean)}")
    print("-" * 65)

    print("\n[1] Class counts (readable images):")
    print(df["class"].value_counts().to_frame(name="Count").to_string())

    print("\n[2] Class counts (clean list used for training):")
    print(clean["class"].value_counts().to_frame(name="Count").to_string())

    print("\n[3] Width, height and file size per class (min / median / max):")
    stats = df.groupby("class")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
    stats.columns = [f"{c[0]}_{c[1]}" for c in stats.columns]
    print(stats.round(1).to_string())

    print("\n[4] Colour modes per class:")
    print(df.groupby(["class", "color_mode"]).size().to_frame(name="Count").to_string())

    print("\n[5] Formats per class:")
    print(df.groupby(["class", "format"]).size().to_frame(name="Count").to_string())

    if corrupt_files:
        print("\n[6] Corrupt files:")
        for c in corrupt_files:
            print(f"  {c['filepath']}  ->  {c['error']}")

    print(line)
    print("Saved: audit_dataset_metadata.csv, audit_corrupt_files.csv,")
    print("       audit_duplicates.csv, clean_file_list.csv")


if __name__ == "__main__":
    if DATA_DIR.exists():
        audit_dataset(DATA_DIR)
    else:
        print(f"Error: '{DATA_DIR}' not found. Extract the Kaggle dataset into './data'.")