import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
DATA_DIR   = "F:\\NTI\\Technical\\Evaluation_tasks\\Flower Classification\\data\\flowers\\flowers"
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 32
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if __name__ == "__main__":

    # ── Load class names ───────────────────────────────────────
    with open("models/class_names.json") as f:
        CLASS_NAMES = json.load(f)
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Classes: {CLASS_NAMES}")

    # ── Recreate same test split as train.py ───────────────────
    full_dataset = datasets.ImageFolder(DATA_DIR)
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    _, _, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    test_ds     = Subset(datasets.ImageFolder(DATA_DIR, transform=val_test_transform), test_idx.indices)
    # num_workers=0 fixes Windows multiprocessing crash
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # ── Load model ─────────────────────────────────────────────
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.\n")

    # ── Run inference on test set ──────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs    = imgs.to(DEVICE)
            outputs = model(imgs)
            preds   = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Print metrics ──────────────────────────────────────────
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"{'='*50}")
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # ── Confusion Matrix Plot ──────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"Confusion Matrix  (Test Acc: {accuracy:.1f}%)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: outputs/confusion_matrix.png")