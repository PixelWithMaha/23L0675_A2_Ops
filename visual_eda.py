import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10})

df = pd.read_csv("audit_dataset_metadata.csv")
classes = df["class"].unique()

fig, axes = plt.subplots(len(classes), 4, figsize=(12, 6))
fig.suptitle("VisionGuard: Representative Image Samples per Class", fontsize=14, fontweight="bold")

for row_idx, cls_name in enumerate(classes):
    cls_df = df[df["class"] == cls_name]
    sample_paths = cls_df["filepath"].sample(n=min(4, len(cls_df)), random_state=42).tolist()
    
    for col_idx, img_path in enumerate(sample_paths):
        ax = axes[row_idx, col_idx] if len(classes) > 1 else axes[col_idx]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")  
        if col_idx == 0:
            ax.set_title(f"Class: {cls_name}", fontsize=11, fontweight="bold", loc="left")

plt.tight_layout()
plt.savefig("sample_images_grid.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved 'sample_images_grid.png'")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.histplot(data=df, x="width", hue="class", kde=True, ax=axes[0], palette="Set2", element="step")
axes[0].set_title("Distribution of Image Width (px)", fontweight="bold")
axes[0].set_xlabel("Width (pixels)")

sns.histplot(data=df, x="height", hue="class", kde=True, ax=axes[1], palette="Set2", element="step")
axes[1].set_title("Distribution of Image Height (px)", fontweight="bold")
axes[1].set_xlabel("Height (pixels)")

sns.histplot(data=df, x="file_size_kb", hue="class", kde=True, ax=axes[2], palette="Set2", element="step")
axes[2].set_title("Distribution of File Size (KB)", fontweight="bold")
axes[2].set_xlabel("File Size (KB)")

plt.tight_layout()
plt.savefig("eda_histograms.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved 'eda_histograms.png'")