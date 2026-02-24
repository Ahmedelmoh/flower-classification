import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
DATA_DIR   = "F:\\NTI\\Technical\\Evaluation_tasks\\Flower Classification\\data\\flowers\\flowers"
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


if __name__ == "__main__":

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    print(f"Using device: {DEVICE}")

    # ── Dataset & Split 70 / 15 / 15 ──────────────────────────
    full_dataset = datasets.ImageFolder(DATA_DIR)
    CLASS_NAMES  = full_dataset.classes
    NUM_CLASSES  = len(CLASS_NAMES)
    print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val

    train_idx, val_idx, test_idx = random_split(
        range(n_total), [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_transform),    train_idx.indices)
    val_ds   = Subset(datasets.ImageFolder(DATA_DIR, transform=val_test_transform), val_idx.indices)
    test_ds  = Subset(datasets.ImageFolder(DATA_DIR, transform=val_test_transform), test_idx.indices)

    # ✅ num_workers=0 fixes the Windows multiprocessing crash
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model: ResNet50, fine-tune layer4 + new head ───────────
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():         # freeze all layers
        param.requires_grad = False

    for param in model.layer4.parameters():  # unfreeze last residual block
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # replace head
    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── Loss / Optimizer / Scheduler ───────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training Loop ───────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    def run_epoch(loader, training=True):
        model.train() if training else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                total      += imgs.size(0)
        return total_loss / total, correct / total

    print("\n" + "="*65)
    print("  Epoch  | Train Loss | Train Acc | Val Loss  | Val Acc")
    print("="*65)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss,   val_acc   = run_epoch(val_loader,   training=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc * 100)
        history["val_acc"].append(val_acc * 100)

        star = " ★" if val_acc > best_val_acc else ""
        print(f"  {epoch:02d}/{EPOCHS}  |  {train_loss:.4f}    |  {train_acc*100:.2f}%   |"
              f"  {val_loss:.4f}   |  {val_acc*100:.2f}%{star}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print("="*65)
    print(f"\n✓ Best Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"✓ Model saved to    : {MODEL_PATH}")

    # ── Plot Training Curves ────────────────────────────────────
    ep = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(ep, history["train_loss"], "b-o", markersize=4, label="Train")
    ax1.plot(ep, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Loss Curve", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, history["train_acc"], "b-o", markersize=4, label="Train")
    ax2.plot(ep, history["val_acc"],   "r-o", markersize=4, label="Val")
    ax2.set_title("Accuracy Curve (%)", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Flower Classification — ResNet50", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: outputs/training_curves.png")

    # Save class names for evaluate.py and predict.py
    with open("models/class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f)
    print("✓ Saved: models/class_names.json")