"""
predict.py — Single-image inference + sample prediction grid.

Usage:
    python predict.py --image path/to/flower.jpg   # predict one image
    python predict.py --samples                     # show grid from test set
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Subset, random_split
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
DATA_DIR   = "F:\\NTI\\Technical\\Evaluation_tasks\\Flower Classification\\data\\flowers\\flowers"
MODEL_PATH = "models/best_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return m.to(DEVICE).eval()


# ─────────────────────────────────────────
#  INFERENCE FUNCTION
# ─────────────────────────────────────────
def predict(image_path: str, model=None) -> dict:
    """
    Accept a single image path.
    Return { 'class': str, 'confidence': float, 'probabilities': dict }
    """
    if model is None:
        model = load_model()

    img  = Image.open(image_path).convert("RGB")
    inp  = transform(img).unsqueeze(0).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        probs = F.softmax(model(inp), dim=1).squeeze()   # [NUM_CLASSES]

    top_idx   = probs.argmax().item()
    top_class = CLASS_NAMES[top_idx]
    top_conf  = probs[top_idx].item() * 100

    all_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
                 for i in range(NUM_CLASSES)}

    return {"class": top_class, "confidence": round(top_conf, 2), "probabilities": all_probs}


# ─────────────────────────────────────────
#  SAMPLE PREDICTIONS GRID
# ─────────────────────────────────────────
def show_sample_predictions(n_samples=12):
    model = load_model()

    full_dataset = datasets.ImageFolder(DATA_DIR)
    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    _, _, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    # No transform — we want raw PIL images to display
    test_ds = Subset(datasets.ImageFolder(DATA_DIR), test_idx.indices)

    indices = random.sample(range(len(test_ds)), n_samples)
    cols    = 4
    rows    = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        img_pil, true_label = test_ds[idx]
        img_path = test_ds.dataset.imgs[test_ds.indices[idx]][0]

        result   = predict(img_path, model)
        pred_cls = result["class"]
        conf     = result["confidence"]
        true_cls = CLASS_NAMES[true_label]
        color    = "green" if pred_cls == true_cls else "red"

        ax.imshow(img_pil)
        ax.set_title(f"Pred: {pred_cls} ({conf:.1f}%)\nTrue: {true_cls}",
                     color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    for ax in axes[n_samples:]:
        ax.axis("off")

    plt.suptitle("Sample Predictions  —  Green = Correct   Red = Wrong",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: outputs/sample_predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--samples", action="store_true", help="Generate sample_predictions.png")
    args = parser.parse_args()

    if args.image:
        result = predict(args.image)
        print(f"\n🌸 Predicted : {result['class']}  ({result['confidence']:.2f}% confidence)")
        print("\nAll class probabilities:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob / 5)
            print(f"  {cls:<12} {prob:5.2f}%  {bar}")

    elif args.samples:
        show_sample_predictions()

    else:
        parser.print_help()