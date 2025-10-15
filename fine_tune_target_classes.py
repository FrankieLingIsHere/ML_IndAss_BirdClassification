"""
CPU-friendly fine-tuning script focused on under-performing classes.

This script:
- Loads the best checkpoint (`results_stage2_accelerated/best_model.pth`).
- Reads per-class accuracies from `final_evaluation_results.json` to identify classes below average.
- Uses `create_data_loaders` from `data_loader.py` to get train/val/test loaders (keeps transforms consistent).
- Builds a `WeightedRandomSampler` that boosts low-performing classes.
- Freezes backbone initially, unfreezes after `unfreeze_at` epochs.
- Tracks train/val loss and accuracy, saves `fine_tune_history.json` and checkpoint `results_stage2_accelerated/best_model_finetuned.pth`.

Run with:
    python fine_tune_target_classes.py

Adjust hyperparameters inside the script as needed for CPU runtime.
"""

import os
import json
import time
import shutil
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Use your project's modules
from models import BirdClassifier
from data_loader import create_data_loaders

# --- Hyperparameters (tweak for CPU) ---
BATCH_SIZE = 8             # small for CPU
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 0            # Windows CPU friendly
EPOCHS = 6                 # small number to keep CPU time reasonable
UNFREEZE_AT = 3            # epoch to start unfreezing backbone
BOOST_FACTOR = 3.0         # how much to upweight underperformers when building sampler/weights
MAX_BOOST = 8.0            # cap per-class boost to avoid extreme weights
LR_HEAD = 5e-5
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
OUT_CKPT = 'results/fine_tune/best_model_finetuned.pth'
HISTORY_PATH = 'results/fine_tune/fine_tune_history.json'
PATIENCE = 3               # early stopping patience (on avg per-class acc)
USE_FOCAL = False          # option to use focal loss
GRAD_ACCUM = 1             # gradient accumulation steps to simulate larger batch size

# Paths
BEST_MODEL_PATH = 'results_stage2_accelerated/best_model.pth'
EVAL_RESULTS_PATH = 'final_evaluation_results.json'
TRAIN_DIR = 'data/Train'
TRAIN_TXT = 'data/train.txt'
TEST_DIR = 'data/Test'
TEST_TXT = 'data/test.txt'

# --- Helpers ---

def load_eval_results(path=EVAL_RESULTS_PATH):
    with open(path, 'r') as f:
        return json.load(f)


def identify_underperformers(eval_results):
    avg = eval_results.get('average_accuracy_per_class')
    per = eval_results.get('per_class_details', {})
    targets = []
    per_class_acc = {}
    for k, v in per.items():
        try:
            acc = float(v['accuracy'])
        except Exception:
            continue
        per_class_acc[int(k)] = acc
        if acc < avg:
            targets.append(int(k))
    return targets, avg, per_class_acc


def build_sampler(dataset, num_classes, targets, boost=BOOST_FACTOR, max_boost=MAX_BOOST):
    """Build a WeightedRandomSampler from various dataset types.

    Supports:
    - torchvision-like datasets with `.samples` (list of (path,label)).
    - datasets with `.targets` (list of labels).
    - torch.utils.data.Subset wrapping one of the above (respects subset indices).
    - fallback: iterate dataset once to collect labels (slower but robust).
    """
    from torch.utils.data import Subset
    base = dataset
    indices = None
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = dataset.indices

    eps = 1e-6
    # Try samples attribute first (typical ImageFolder-like)
    if hasattr(base, 'samples'):
        samples = base.samples
        if indices is not None:
            samples = [samples[i] for i in indices]
        counts = [0] * num_classes
        for _, label in samples:
            counts[int(label)] += 1
        base_weights = [1.0 / (c + eps) for c in counts]
        sample_weights = [base_weights[int(label)] * (min(max_boost, boost) if int(label) in targets else 1.0) for _, label in samples]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Next try .targets (some datasets expose this)
    if hasattr(base, 'targets'):
        targets_list = base.targets
        if indices is not None:
            targets_list = [targets_list[i] for i in indices]
        counts = [0] * num_classes
        for label in targets_list:
            counts[int(label)] += 1
        base_weights = [1.0 / (c + eps) for c in counts]
        sample_weights = [base_weights[int(label)] * (min(max_boost, boost) if int(label) in targets else 1.0) for label in targets_list]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Fallback: iterate the dataset once to collect labels (works for custom datasets)
    labels = []
    try:
        for item in base:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                labels.append(int(item[1]))
    except Exception:
        # As last resort, raise a clear error
        raise RuntimeError('Unable to build sampler: dataset does not expose samples/targets and is not iterable in expected (img,label) form')

    if indices is not None:
        # If base is iterable but subset provided, filter by indices (slow)
        labels = [labels[i] for i in indices]
    counts = [0] * num_classes
    for label in labels:
        counts[int(label)] += 1
    base_weights = [1.0 / (c + eps) for c in counts]
    sample_weights = [base_weights[int(label)] * (min(max_boost, boost) if int(label) in targets else 1.0) for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# --- Training/Eval loops ---

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total * 100.0


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                class_total[t] += 1
                if p == t:
                    class_correct[t] += 1
    per_class_acc = [(class_correct[i] / class_total[i] * 100.0) if class_total[i] > 0 else 0.0 for i in range(num_classes)]
    return running_loss / total, correct / total * 100.0, per_class_acc


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


# --- Main ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--head-lr', type=float, default=LR_HEAD)
    parser.add_argument('--backbone-lr', type=float, default=LR_BACKBONE)
    parser.add_argument('--boost', type=float, default=BOOST_FACTOR)
    parser.add_argument('--focal', action='store_true', help='Use focal loss')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--out-dir', type=str, default='./results/fine_tune', help='Directory to save checkpoints')
    parser.add_argument('--grad-accum', type=int, default=GRAD_ACCUM, help='Gradient accumulation steps')
    parser.add_argument('--no-early-stop', action='store_true', help='Disable early stopping based on avg per-class accuracy')
    parser.add_argument('--allow-test-selection', action='store_true', help='(Unsafe) enable model selection using Test metrics (will evaluate test each epoch and save test-best checkpoints)')
    parser.add_argument('--final-mode', type=str, default='topk_avg', choices=['topk_avg','swa','last'], help='How to produce final single checkpoint: topk_avg (average top-K val epochs), swa (average last-N epochs), last (final epoch)')
    parser.add_argument('--final-k', type=int, default=5, help='K for topk_avg or N for swa (number of checkpoints to average)')
    parser.add_argument('--save-final-name', type=str, default='final_model_averaged.pth', help='Filename for the final produced checkpoint')
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR_HEAD = args.head_lr
    LR_BACKBONE = args.backbone_lr
    BOOST_FACTOR = args.boost
    USE_FOCAL = args.focal
    RESUME_CKPT = args.resume
    OUT_DIR = args.out_dir
    GRAD_ACCUM = args.grad_accum
    ENABLE_EARLY_STOP = not args.no_early_stop
    ALLOW_TEST_SELECTION = args.allow_test_selection
    FINAL_MODE = args.final_mode
    FINAL_K = args.final_k
    FINAL_NAME = args.save_final_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # If user provided a resume checkpoint, use it; otherwise require the default BEST_MODEL_PATH
    if RESUME_CKPT:
        BEST_MODEL_PATH = RESUME_CKPT
    else:
        if not os.path.exists(BEST_MODEL_PATH):
            raise AssertionError('Best model not found: {}'.format(BEST_MODEL_PATH))

    # Load evaluation results if present; otherwise fall back to no targeted boosting
    if os.path.exists(EVAL_RESULTS_PATH):
        eval_results = load_eval_results(EVAL_RESULTS_PATH)
        targets, avg_acc, per_class_acc = identify_underperformers(eval_results)
    else:
        print('Warning: evaluation results not found at {}. Proceeding without underperformer targeting.'.format(EVAL_RESULTS_PATH))
        targets = []
        avg_acc = 0.0
        per_class_acc = {}
    print('Identified {} under-performing classes (avg={:.2f}%)'.format(len(targets), avg_acc))

    # Build data loaders using your helper to keep transforms identical
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        TRAIN_DIR, TRAIN_TXT, TEST_DIR, TEST_TXT,
        batch_size=BATCH_SIZE, image_size=384, num_workers=NUM_WORKERS, validation_split=0.1, augmentation_level='advanced'
    )

    # Build sampler on the underlying full train dataset object in data_loader
    # Prefer using the dataset object returned by the train_loader (keeps transforms identical).
    # If that's not available, recreate the BirdDataset with the project's training transform so
    # the DataLoader yields tensors (not PIL images).
    if hasattr(train_loader, 'dataset'):
        full_train_dataset = train_loader.dataset
    else:
        from data_loader import BirdDataset, get_data_transforms
        train_transform = get_data_transforms(image_size=384, is_training=True, augmentation_level='advanced')
        full_train_dataset = BirdDataset(TRAIN_DIR, TRAIN_TXT, transform=train_transform)
    sampler = build_sampler(full_train_dataset, num_classes, targets, boost=BOOST_FACTOR)

    # Replace train_loader with sampler-based loader to oversample targets
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)

    # Load model and checkpoint
    # Use EfficientNet-B4 by default for stronger representational power on Colab/GPU.
    model = BirdClassifier(num_classes=num_classes, architecture='efficientnet_b4', pretrained=False, dropout_rate=0.3)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    # Handle two checkpoint formats
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and all(k.startswith('module.') for k in ckpt.keys()):
        state = {k.replace('module.', ''): v for k, v in ckpt.items()}
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)

    # Freeze backbone initially
    for name, p in model.named_parameters():
        if 'backbone' in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    # Loss and optimizer (class weights: dynamic based on deficit from avg)
    class_weights = torch.ones(num_classes, device=device)
    # compute per-class multipliers from per_class_acc where available
    for c in range(num_classes):
        if c in per_class_acc:
            deficit = max(0.0, avg_acc - per_class_acc[c])
            # normalize deficit by avg_acc and scale by BOOST_FACTOR
            mult = 1.0 + (deficit / max(1e-6, avg_acc)) * BOOST_FACTOR
            class_weights[c] = min(MAX_BOOST, float(mult))
        elif c in targets:
            class_weights[c] = min(MAX_BOOST, float(BOOST_FACTOR))
    if USE_FOCAL:
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': LR_BACKBONE},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': LR_HEAD}
        ], weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val = -1.0
    best_avg_class = -1.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'per_class_acc': []}
    # optional test metrics (filled only if ALLOW_TEST_SELECTION)
    history['test_loss'] = []
    history['test_acc'] = []
    history['test_per_class_acc'] = []

    # bookkeeping for optional test-based selection
    best_test_top1 = -1.0
    best_test_avgclass = -1.0
    if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION:
        print('WARNING: Test-based checkpoint selection is ENABLED. This may introduce evaluation leakage. Proceed only if you understand the implications.')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, criterion, device, num_classes)

        # Optional Test evaluation (explicit opt-in)
        if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION:
            test_loss, test_acc, test_per_class = evaluate(model, test_loader, criterion, device, num_classes)
        else:
            test_loss, test_acc, test_per_class = (None, None, None)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['per_class_acc'].append(per_class_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_per_class_acc'].append(test_per_class)

        print('Epoch {}/{} - train_loss: {:.4f} train_acc: {:.2f}% | val_loss: {:.4f} val_acc: {:.2f}% ({:.1f}s)'.format(
            epoch, EPOCHS, train_loss, train_acc, val_loss, val_acc, time.time()-t0))

        # Save best by val_acc when it improves
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, OUT_CKPT)
            print(' -> New best saved (val_acc {:.2f}%)'.format(val_acc))

        # If enabled, save best by test top-1 (explicit opt-in)
        if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION and test_acc is not None:
            if test_acc > best_test_top1 + 1e-4:
                best_test_top1 = test_acc
                tk = os.path.join(OUT_DIR, 'best_model_test_top1.pth')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'test_acc': test_acc}, tk)
                print(' -> New test-top1 best saved (test_acc {:.2f}%) -> {}'.format(test_acc, tk))

        # Save best by average per-class accuracy as primary objective
        avg_per_class = sum(per_class_acc) / len(per_class_acc)
        if avg_per_class > best_avg_class + 1e-4:
            best_avg_class = avg_per_class
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'avg_class_acc': avg_per_class}, OUT_CKPT.replace('.pth', '_avgclass.pth'))
            print(' -> New best saved by avg-class-acc ({:.2f}%)'.format(avg_per_class))

        # If enabled, save best by test avg-class (explicit opt-in)
        if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION and test_per_class is not None:
            test_avg_class = sum(test_per_class) / len(test_per_class)
            if test_avg_class > best_test_avgclass + 1e-4:
                best_test_avgclass = test_avg_class
                tk2 = os.path.join(OUT_DIR, 'best_model_test_avgclass.pth')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'test_avg_class': test_avg_class}, tk2)
                print(' -> New test-avgclass best saved ({:.2f}%) -> {}'.format(test_avg_class, tk2))

    # Gradual unfreeze: start unfreezing parts of backbone at UNFREEZE_AT, full at UNFREEZE_AT+1
        if epoch == UNFREEZE_AT:
            print('Gradually unfreezing last stages of backbone...')
            # Unfreeze parameters whose name contains 'layer4' or last blocks if ResNet-like
            for name, p in model.named_parameters():
                if 'backbone' in name and ('layer4' in name or 'block5' in name or 'conv_head' in name):
                    p.requires_grad = True
        if epoch == UNFREEZE_AT + 1:
            print('Unfreezing full backbone for deeper fine-tuning...')
            for name, p in model.named_parameters():
                p.requires_grad = True
            # Rebuild optimizer with differential LR
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': LR_BACKBONE},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': LR_HEAD}
            ], weight_decay=WEIGHT_DECAY)

        # Step scheduler based on avg per-class accuracy
        scheduler.step(avg_per_class)
        # Save per-epoch checkpoint (keeps a record for later averaging)
        os.makedirs(OUT_DIR, exist_ok=True)
        epoch_ckpt = os.path.join(OUT_DIR, f'ckpt_epoch_{epoch}.pth')
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc, 'avg_class_acc': avg_per_class}, epoch_ckpt)

        # Early stopping based on avg per-class accuracy (optional)
        if ENABLE_EARLY_STOP:
            if avg_per_class <= best_avg_class + 1e-4:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0

            if epochs_no_improve >= PATIENCE:
                print('Early stopping: no improvement in avg per-class acc for {} epochs'.format(PATIENCE))
                break

    # Save history
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    # --- Final checkpoint creation (no test leakage) ---
    def average_state_dicts(paths):
        # loads state_dicts from given paths and returns an averaged state_dict
        avg_state = {}
        count = 0
        for p in paths:
            ck = torch.load(p, map_location='cpu')
            if isinstance(ck, dict) and 'state_dict' in ck:
                sd = ck['state_dict']
            else:
                sd = ck
            if not avg_state:
                # initialize with zero tensors of same shape
                for k, v in sd.items():
                    avg_state[k] = v.clone().float()
                count = 1
            else:
                for k, v in sd.items():
                    if k in avg_state:
                        avg_state[k] += v.clone().float()
                count += 1
        # average
        for k in avg_state:
            avg_state[k] = (avg_state[k] / float(count)).type(sd[k].dtype)
        return avg_state

    # Choose final checkpoints according to FINAL_MODE
    all_ckpts = []
    for e_idx in range(1, len(history['val_acc']) + 1):
        p = os.path.join(OUT_DIR, f'ckpt_epoch_{e_idx}.pth')
        if os.path.exists(p):
            all_ckpts.append((e_idx, p))

    final_ckpt_path = os.path.join(OUT_DIR, FINAL_NAME)
    if FINAL_MODE == 'last':
        # use last epoch checkpoint
        if all_ckpts:
            shutil.copy(all_ckpts[-1][1], final_ckpt_path)
            print('Final checkpoint (last) written to', final_ckpt_path)
    elif FINAL_MODE == 'swa':
        # average last FINAL_K checkpoints
        selected = [p for _, p in all_ckpts[-FINAL_K:]] if all_ckpts else []
        if selected:
            avg_sd = average_state_dicts(selected)
            torch.save({'state_dict': avg_sd}, final_ckpt_path)
            print('Final checkpoint (SWA last {}) written to {}'.format(FINAL_K, final_ckpt_path))
    else:  # topk_avg
        # compute avg-per-class per epoch from history and pick top-K epochs
        per_epoch_avgclass = [sum(pc)/len(pc) if pc else 0.0 for pc in history['per_class_acc']]
        ranked = sorted(list(enumerate(per_epoch_avgclass, start=1)), key=lambda x: x[1], reverse=True)
        selected_epochs = [e for e, _ in ranked[:FINAL_K]]
        selected_paths = [os.path.join(OUT_DIR, f'ckpt_epoch_{e}.pth') for e in selected_epochs if os.path.exists(os.path.join(OUT_DIR, f'ckpt_epoch_{e}.pth'))]
        if selected_paths:
            avg_sd = average_state_dicts(selected_paths)
            torch.save({'state_dict': avg_sd}, final_ckpt_path)
            print('Final checkpoint (top-{} avg by val avg-class) written to {}'.format(FINAL_K, final_ckpt_path))
        else:
            print('No per-epoch checkpoints found to average; no final averaged checkpoint created.')

    print('Fine-tuning complete. Best val acc: {:.2f}%'.format(best_val))
    print('Checkpoint saved to:', OUT_CKPT)

