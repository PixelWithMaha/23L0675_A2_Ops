import matplotlib.pyplot as plt
from src.dataset import get_data_splits

def save_augmented_batch_figure():
    train_ds, _, _, class_names = get_data_splits("data")
    
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(9, 9))
        for i in range(min(9, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            idx = int(labels[i].numpy())
            plt.title(class_names[idx], fontsize=10)
            plt.axis("off")
            
        plt.tight_layout()
        plt.savefig("augmented_batch_sample.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    save_augmented_batch_figure()