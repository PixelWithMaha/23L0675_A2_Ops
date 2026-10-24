import os
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def audit_dataset(dataset_dir):
    data_path = Path(dataset_dir)
    records = []
    seen_hashes = {}
    duplicate_records = []
    corrupt_files = []

    print(f"Auditing dataset at: {data_path.resolve()}...\n")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for filepath in data_path.rglob("*"):
        if filepath.suffix.lower() not in valid_extensions:
            continue

        class_name = filepath.parent.name
        file_size_bytes = os.path.getsize(filepath)

        try:
            with Image.open(filepath) as img:
                img.verify()
            
            with Image.open(filepath) as img:
                width, height = img.size
                color_mode = img.mode
                img_format = img.format

            content_hash = compute_sha256(filepath)

            if content_hash in seen_hashes:
                duplicate_records.append({
                    "duplicate_file": str(filepath),
                    "original_file": seen_hashes[content_hash],
                    "sha256": content_hash
                })
            else:
                seen_hashes[content_hash] = str(filepath)

            records.append({
                "filepath": str(filepath),
                "class": class_name,
                "file_size_kb": file_size_bytes / 1024.0,
                "width": width,
                "height": height,
                "color_mode": color_mode,
                "format": img_format,
                "sha256": content_hash
            })

        except Exception as err:
            corrupt_files.append({"filepath": str(filepath), "error": str(err)})

    df = pd.DataFrame(records)

    df.to_csv("audit_dataset_metadata.csv", index=False)
    
    if duplicate_records:
        pd.DataFrame(duplicate_records).to_csv("dataset_duplicates.csv", index=False)

    print("=" * 65)
    print("TASK 1.2: DATASET AUDIT SUMMARY REPORT")
    print("=" * 65)
    print(f"Total Valid Images Processed : {len(df)}")
    print(f"Corrupt / Unreadable Files   : {len(corrupt_files)}")
    print(f"Exact Duplicate Files        : {len(duplicate_records)}")
    print("-" * 65)
    
    print("\n[1] Class Counts Table:")
    print(df["class"].value_counts().to_frame(name="Count").to_string())

    print("\n[2] Summary Statistics per Class (Min, Median, Max):")
    stats = df.groupby("class")[["width", "height", "file_size_kb"]].agg(["min", "median", "max"])
   
    stats.columns = [f"{col[0]}_{col[1]}" for col in stats.columns]
    print(stats.to_string())

    print("\n[3] Color Modes Breakdown:")
    print(df.groupby(["class", "color_mode"]).size().to_frame(name="Count").to_string())

    print("\n[4] Formats Breakdown:")
    print(df.groupby(["class", "format"]).size().to_frame(name="Count").to_string())
    print("=" * 65)
    print("Full metadata table saved to 'audit_dataset_metadata.csv'")

if __name__ == "__main__":
    DATASET_PATH = "./data"
    if os.path.exists(DATASET_PATH):
        audit_dataset(DATASET_PATH)
    else:
        print(f"Error: '{DATASET_PATH}' directory not found. Extract your Kaggle dataset into './data'.")