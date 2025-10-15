import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

# === ADJUST PATHS / HYPERPARAMS ===
MODEL_PATH = "results_stage2_accelerated/best_model_finetuned.pth"   # or best_model.pth
TRAIN_ANNOT = "data/train.txt"
TRAIN_ROOT = "data/Train"
IMG_SIZE = 384
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ================================

# Attempt to import your dataset and transforms
try:
    from data_loader import BirdDataset, get_data_transforms
except Exception as e:
    raise RuntimeError("Adjust imports: ensure `data_loader.BirdDataset` and `get_data_transforms` are available: {}".format(e))

# Try to import model factory - adjust to your constructor if needed
try:
    from models import BirdClassifier
except Exception:
    BirdClassifier = None

# Build deterministic transforms (same as validation)
try:
    val_transform = get_data_transforms(IMG_SIZE, is_training=False)
except TypeError:
    # fallback if api differs: try (image_size, False, 'none')
    val_transform = get_data_transforms(IMG_SIZE, False, "none")

# Build training dataset with deterministic transforms
# BirdDataset signature: BirdDataset(image_dir, annotation_file, transform=None)
train_dataset = BirdDataset(TRAIN_ROOT, TRAIN_ANNOT, transform=val_transform)

# Derive num_classes
if hasattr(train_dataset, "classes"):
    num_classes = len(train_dataset.classes)
else:
    # fallback: infer from labels in the annotation file
    labels = [int(line.strip().split()[1]) for line in open(TRAIN_ANNOT)]
    num_classes = max(labels) + 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Load model
if BirdClassifier is not None:
    # Instantiate with EfficientNet-B3 to match the checkpoint used during fine-tuning
    try:
        model = BirdClassifier(num_classes=num_classes, architecture='efficientnet_b3', pretrained=False, dropout_rate=0.3)
    except Exception:
        # Fallback to default constructor
        model = BirdClassifier(num_classes=num_classes)
else:
    raise RuntimeError("Please instantiate your model here (adjust code to match your models.py).")

# Load checkpoint robustly
ckpt = torch.load(MODEL_PATH, map_location="cpu")
if isinstance(ckpt, dict) and ("state_dict" in ckpt or "model_state" in ckpt):
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt.get("model_state", ckpt)
else:
    state = ckpt

# strip DataParallel 'module.' if present
new_state = {}
for k, v in state.items():
    new_k = k.replace("module.", "") if k.startswith("module.") else k
    new_state[new_k] = v

print("Loaded checkpoint:", MODEL_PATH)
try:
    res = model.load_state_dict(new_state, strict=False)
    # PyTorch versions differ on return type of load_state_dict
    try:
        # res may be a NamedTuple with missing_keys/unexpected_keys
        missing = getattr(res, 'missing_keys', None)
        unexpected = getattr(res, 'unexpected_keys', None)
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    except Exception:
        # If res is a simple dict or None, just print type
        print("load_state_dict returned:", type(res))
except Exception as e:
    print("Error loading state_dict:", e)

model.to(DEVICE)
model.eval()

# Run evaluation on dataset (deterministic)
total = 0
correct = 0
per_class_total = [0] * num_classes
per_class_correct = [0] * num_classes

with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            per_class_total[t] += 1
            if t == p:
                per_class_correct[t] += 1

overall_acc = 100.0 * correct / total if total > 0 else 0.0
avg_class_acc = 100.0 * sum(( (pc/pt) if pt>0 else 0) for pc,pt in zip(per_class_correct, per_class_total))/num_classes

print("Clean-train (eval-mode, no-aug) overall accuracy: {:.2f}%".format(overall_acc))
print("Clean-train average per-class accuracy: {:.2f}%".format(avg_class_acc))
print("Samples evaluated: {}".format(total))
# Optionally print worst classes
worst = sorted([(i, per_class_total[i], per_class_correct[i]) for i in range(num_classes)], key=lambda x: (0 if x[1]==0 else x[2]/x[1]))
print("Worst 10 classes (class_id, total, correct): {}".format(worst[:10]))
