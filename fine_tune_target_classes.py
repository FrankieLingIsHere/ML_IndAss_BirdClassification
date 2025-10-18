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
import logging
import subprocess
from collections.abc import Mapping
import re
import random
import numpy as np

# Use your project's modules
from models import BirdClassifier
from data_loader import create_data_loaders
from training_utils import freeze_batchnorm_stats, unfreeze_batchnorm_stats, gradual_unfreeze
from trainer import calculate_class_weights

# --- Hyperparameters (tweak for CPU) ---
BATCH_SIZE = 8             # small for CPU
EVAL_BATCH_SIZE = 16
NUM_WORKERS = 0            # Windows CPU friendly
EPOCHS = 6                 # small number to keep CPU time reasonable
UNFREEZE_AT = 3            # (deprecated) epoch to start unfreezing backbone
# New schedule controls: freeze for N epochs, then gradually unfreeze for M epochs, then fully unfrozen
FREEZE_EPOCHS = 10        # default: keep backbone frozen for 10 epochs
GRADUAL_UNFREEZE_EPOCHS = 5  # default: spread unfreezing across 5 epochs
BOOST_FACTOR = 3.0         # how much to upweight underperformers when building sampler/weights
MAX_BOOST = 8.0            # cap per-class boost to avoid extreme weights
LR_HEAD = 5e-5
LR_BACKBONE = 1e-5
LR_BACKBONE_UNFREEZE = LR_BACKBONE * 0.1
WEIGHT_DECAY = 1e-4
OUT_CKPT = 'results/fine_tune/best_model_finetuned.pth'
HISTORY_PATH = 'results/fine_tune/fine_tune_history.json'
PATIENCE = 3               # early stopping patience (on avg per-class acc)
USE_FOCAL = False          # option to use focal loss
GRAD_ACCUM = 1             # gradient accumulation steps to simulate larger batch size

# Paths
BEST_MODEL_PATH = 'results/best_model.pth'
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


def detect_backbone_block_count(model):
    """Heuristic detection of backbone block count for adaptive unfreeze scheduling."""
    try:
        keys = list(model.state_dict().keys())
    except Exception:
        try:
            keys = [k for k, _ in model.named_parameters()]
        except Exception:
            return 5
    # EfficientNet-like: _blocks.<idx>.
    idxs = set()
    for k in keys:
        m = re.search(r'_blocks\.(\d+)\.', k)
        if m:
            idxs.add(int(m.group(1)))
    if idxs:
        return max(idxs) + 1
    # ResNet-like: layer<idx>.
    idxs2 = set()
    for k in keys:
        m2 = re.search(r'layer(\d+)\.', k)
        if m2:
            idxs2.add(int(m2.group(1)))
    if idxs2:
        return max(idxs2) + 1
    # Fallback: count 'block' occurrences and estimate
    blocks = [k for k in keys if 'block' in k or 'blocks' in k]
    if blocks:
        # heuristically map number of block-like keys to block count
        return max(1, min(20, len(blocks) // 8))
    return 5

# --- Training/Eval loops ---

def train_one_epoch(model, loader, criterion, optimizer, device, grad_accum=1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device); labels = labels.to(device)
        # Apply CutMix or Mixup if configured; CutMix takes precedence
        try:
            ca = float(globals().get('CUTMIX_ALPHA', 0.0))
        except Exception:
            ca = 0.0
        try:
            ma = float(globals().get('MIXUP_ALPHA', 0.0))
        except Exception:
            ma = 0.0
        mixup_active = False
        cutmix_active = False
        # Decide whether to apply CutMix for this batch based on CUTMIX_PROB
        apply_cutmix = False
        try:
            apply_cutmix = (random.random() < float(globals().get('CUTMIX_PROB', 1.0)))
        except Exception:
            apply_cutmix = True
        if ca and ca > 0.0 and apply_cutmix:
            # CutMix: sample lambda and create a random box
            try:
                lam = np.random.beta(ca, ca)
            except Exception:
                import numpy as np
                lam = float(np.random.beta(ca, ca))
            batch_size = imgs.size(0)
            idx = torch.randperm(batch_size)
            imgs_shuf = imgs[idx]
            labels_shuf = labels[idx]
            # compute bbox
            W = imgs.size(3)
            H = imgs.size(2)
            cut_rat = math.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            # uniform center
            cx = torch.randint(0, W, (1,)).item()
            cy = torch.randint(0, H, (1,)).item()
            x1 = max(0, cx - cut_w // 2)
            y1 = max(0, cy - cut_h // 2)
            x2 = min(W, cx + cut_w // 2)
            y2 = min(H, cy + cut_h // 2)
            imgs[:, :, y1:y2, x1:x2] = imgs_shuf[:, :, y1:y2, x1:x2]
            # adjust lambda to exactly match pixel ratio
            box_area = float((y2 - y1) * (x2 - x1))
            lam = 1.0 - box_area / float(H * W)
            cutmix_active = True
        elif ma and ma > 0.0:
            try:
                lam = np.random.beta(ma, ma)
            except Exception:
                import numpy as np
                lam = float(np.random.beta(ma, ma))
            # create shuffled pairings
            idx = torch.randperm(imgs.size(0))
            imgs_shuf = imgs[idx]
            labels_shuf = labels[idx]
            imgs = imgs * lam + imgs_shuf * (1.0 - lam)
            mixup_active = True
        outputs = model(imgs)
        # Handle HF transformers ModelOutput or tuple returns (extract logits)
        try:
            if not isinstance(outputs, torch.Tensor):
                # HF returns ModelOutput with .logits
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                # some models return (logits, ) tuples
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                    outputs = outputs[0]
                # dict-like
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    outputs = outputs['logits']
        except Exception:
            pass
        if cutmix_active or mixup_active:
            # mix loss for cutmix or mixup: lam * loss(orig) + (1-lam) * loss(shuffled)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(outputs, labels_shuf)
            loss = lam * loss1 + (1.0 - lam) * loss2
        else:
            loss = criterion(outputs, labels)
        # scale loss for gradient accumulation
        (loss / float(grad_accum)).backward()
        # step and zero every grad_accum steps
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    # if there are leftover gradients (when len(loader) % grad_accum != 0)
    if (step + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
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
            # Handle HF transformers ModelOutput or tuple returns (extract logits)
            try:
                if not isinstance(outputs, torch.Tensor):
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                        outputs = outputs[0]
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        outputs = outputs['logits']
            except Exception:
                pass
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
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--unfreeze-at', type=int, default=UNFREEZE_AT, help='(deprecated) Epoch to start gradual unfreeze')
    parser.add_argument('--freeze-epochs', type=int, default=FREEZE_EPOCHS, help='Number of initial epochs to keep backbone fully frozen')
    parser.add_argument('--gradual-unfreeze-epochs', type=int, default=GRADUAL_UNFREEZE_EPOCHS, help='Number of epochs over which to gradually unfreeze the backbone')
    parser.add_argument('--backbone-lr-unfreeze', type=float, default=None, help='Backbone LR to use when fully unfrozen (default: 0.1 * backbone-lr)')
    parser.add_argument('--auto-select-best', action='store_true', help='Automatically select best epoch checkpoint (by val_acc) and copy it to OUT_CKPT at the end')
    parser.add_argument('--select-based-on', type=str, choices=['val_avg_class', 'test_avg_class'], default='val_avg_class', help='Which metric to use for auto-selecting the best epoch when --auto-select-best is enabled')
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='Enable mixup with given alpha (Beta distribution). 0 disables mixup.')
    parser.add_argument('--cutmix-alpha', type=float, default=0.0, help='Enable CutMix with given alpha. 0 disables CutMix')
    parser.add_argument('--randaugment', action='store_true', help='Enable RandAugment on training transforms (if available in torchvision)')
    parser.add_argument('--cutmix-prob', type=float, default=1.0, help='Probability of applying CutMix to a given batch (0-1)')
    parser.add_argument('--use-class-weights', action='store_true', help='Compute class weights from training set frequencies and use them in the loss')
    parser.add_argument('--oversample', action='store_true', help='Oversample all classes by inverse-frequency (WeightedRandomSampler) instead of only boosting underperformers')
    parser.add_argument('--soften-augment', action='store_true', help='Temporarily reduce strong augmentations (lower cutmix prob, lower mixup alpha) for safer fine-tuning')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    parser.add_argument('--focal-alpha', type=float, default=1.0, help='Alpha (scaling) parameter for focal loss')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Early stopping patience (epochs)')
    parser.add_argument('--best-model-path', type=str, default=BEST_MODEL_PATH, help='Path to base checkpoint to resume from')
    parser.add_argument('--head-lr', type=float, default=LR_HEAD)
    parser.add_argument('--backbone-lr', type=float, default=LR_BACKBONE)
    parser.add_argument('--boost', type=float, default=BOOST_FACTOR)
    parser.add_argument('--focal', action='store_true', help='Use focal loss')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--out-dir', type=str, default='./results/fine_tune', help='Directory to save checkpoints')
    parser.add_argument('--grad-accum', type=int, default=GRAD_ACCUM, help='Gradient accumulation steps')
    parser.add_argument('--early-stop', action='store_true', help='Enable early stopping based on avg per-class accuracy')
    parser.add_argument('--allow-test-selection', action='store_true', help='(Unsafe) enable model selection using Test metrics (will evaluate test each epoch and save test-best checkpoints)')
    parser.add_argument('--eval-split', type=str, choices=['val','test'], default='val', help='Which split to use for avg-per-class evaluation and primary checkpointing during training')
    parser.add_argument('--final-mode', type=str, default='topk_avg', choices=['topk_avg','swa','last'], help='How to produce final single checkpoint: topk_avg (average top-K val epochs), swa (average last-N epochs), last (final epoch)')
    parser.add_argument('--final-k', type=int, default=5, help='K for topk_avg or N for swa (number of checkpoints to average)')
    parser.add_argument('--save-final-name', type=str, default='final_model_averaged.pth', help='Filename for the final produced checkpoint')
    parser.add_argument('--from-scratch', action='store_true', help='Train from random initialization (do not load any checkpoint)')
    parser.add_argument('--hf-model', type=str, default=None, help='HuggingFace model id to download and inject backbone weights (e.g. chriamue/bird-species-classifier)')
    parser.add_argument('--strict-inject', action='store_true', help='When injecting HF weights, fail if less than 50% of backbone keys matched')
    parser.add_argument('--use-hf-direct', action='store_true', help='Use HF AutoModelForImageClassification object directly (replace local BirdClassifier). Helpful when HF backbone architecture differs from local backbone, e.g. Swin.)')
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR_HEAD = args.head_lr
    LR_BACKBONE = args.backbone_lr
    BOOST_FACTOR = args.boost
    USE_FOCAL = args.focal
    USE_CLASS_WEIGHTS = args.use_class_weights
    OVERSAMPLE_ALL = args.oversample
    SOFTEN_AUG = args.soften_augment
    FOCAL_GAMMA = args.focal_gamma
    FOCAL_ALPHA = args.focal_alpha
    RESUME_CKPT = args.resume
    OUT_DIR = args.out_dir
    GRAD_ACCUM = args.grad_accum
    ENABLE_EARLY_STOP = args.early_stop
    ALLOW_TEST_SELECTION = args.allow_test_selection
    FINAL_MODE = args.final_mode
    FINAL_K = args.final_k
    FINAL_NAME = args.save_final_name
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    NUM_WORKERS = args.num_workers
    UNFREEZE_AT = args.unfreeze_at
    FREEZE_EPOCHS = args.freeze_epochs
    GRADUAL_UNFREEZE_EPOCHS = args.gradual_unfreeze_epochs
    BACKBONE_LR_UNFREEZE = args.backbone_lr_unfreeze if args.backbone_lr_unfreeze is not None else LR_BACKBONE_UNFREEZE
    PATIENCE = args.patience
    BEST_MODEL_PATH = args.best_model_path
    FROM_SCRATCH = args.from_scratch
    HF_MODEL_ID = args.hf_model
    STRICT_INJECT = args.strict_inject
    USE_HF_DIRECT = args.use_hf_direct
    AUTO_SELECT_BEST = args.auto_select_best
    SELECT_BASED_ON = args.select_based_on
    MIXUP_ALPHA = args.mixup_alpha
    CUTMIX_ALPHA = args.cutmix_alpha
    USE_RANDAUG = args.randaugment
    CUTMIX_PROB = args.cutmix_prob
    SEED = args.seed
    EVAL_SPLIT = args.eval_split

    # Validation: prevent confusing mixed initialization modes
    if USE_HF_DIRECT and RESUME_CKPT:
        raise RuntimeError('Incompatible options: --use-hf-direct replaces the model with a HuggingFace model object, '
                           'while --resume attempts to load a checkpoint into the model.\n'
                           'If you want to resume your previously trained local checkpoint, omit --use-hf-direct.\n'
                           'If you want to use the HF model instead, omit --resume.')

    # Ensure output directory exists and update derived paths
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_CKPT = os.path.join(OUT_DIR, 'best_model_finetuned.pth')
    HISTORY_PATH = os.path.join(OUT_DIR, 'fine_tune_history.json')

    # --- Logging / Run config snapshot ---
    logger = logging.getLogger('fine_tune')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(OUT_DIR, 'run.log'))
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

    # Save a small run config for reproducibility
    run_config = {
        'args': vars(args),
        'torch_version': torch.__version__,
    }
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        run_config['git_sha'] = git_sha
    except Exception:
        run_config['git_sha'] = None
    with open(os.path.join(OUT_DIR, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)
    logger.info('Run config written to %s', os.path.join(OUT_DIR, 'run_config.json'))

    # Seed RNGs for reproducibility when requested
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        try:
            # guard in case torch.backends.cudnn is not present in some minimal builds
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print(f'Backbone schedule: freeze {FREEZE_EPOCHS} epochs, gradual unfreeze {GRADUAL_UNFREEZE_EPOCHS} epochs, then fully unfrozen; early stopping patience {PATIENCE}')



    # Check loading mode: from-scratch, HF model injection, or resume from checkpoint
    if FROM_SCRATCH:
        logger.info('Running from scratch; no checkpoint will be loaded')
    else:
        # prefer explicit resume path
        if RESUME_CKPT:
            BEST_MODEL_PATH = RESUME_CKPT
        # If HF model id provided, we'll inject its weights (below)
        elif HF_MODEL_ID is None:
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
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS, validation_split=0.1, augmentation_level='advanced', use_randaugment=USE_RANDAUG
    )

    # Build sampler on the underlying full train dataset object in data_loader
    # Prefer using the dataset object returned by the train_loader (keeps transforms identical).
    # If that's not available, recreate the BirdDataset with the project's training transform so
    # the DataLoader yields tensors (not PIL images).
    if hasattr(train_loader, 'dataset'):
        full_train_dataset = train_loader.dataset
    else:
        from data_loader import BirdDataset, get_data_transforms
        train_transform = get_data_transforms(image_size=IMAGE_SIZE, is_training=True, augmentation_level='advanced')
        full_train_dataset = BirdDataset(TRAIN_DIR, TRAIN_TXT, transform=train_transform)
    # If oversample-all requested, set targets to all classes so build_sampler will upweight all rare classes
    sampler_targets = targets
    if OVERSAMPLE_ALL:
        sampler_targets = list(range(num_classes))
    sampler = build_sampler(full_train_dataset, num_classes, sampler_targets, boost=BOOST_FACTOR)

    # Replace train_loader with sampler-based loader to oversample targets
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    # Adjust val/test batch sizes for evaluation speed
    # If the original val/test loaders exist, prefer their dataset and transforms
    if hasattr(val_loader, 'dataset'):
        val_dataset = val_loader.dataset
        val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    if hasattr(test_loader, 'dataset'):
        test_dataset = test_loader.dataset
        test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load model and optionally load/pre-inject checkpoint/backbone
    # By default we build the project's BirdClassifier (EfficientNet-B4). However,
    # when --use-hf-direct is specified we will replace this with the HF
    # AutoModelForImageClassification object (useful for Swin/ViT models from HF).
    model = BirdClassifier(num_classes=num_classes, architecture='efficientnet_b4', pretrained=False, dropout_rate=0.3)
    model_sd = model.state_dict()

    # Detect backbone block count to adapt gradual-unfreeze mapping (safe default on failure)
    try:
        TOTAL_BACKBONE_BLOCKS = detect_backbone_block_count(model)
    except Exception:
        TOTAL_BACKBONE_BLOCKS = 5
    print(f'Detected {TOTAL_BACKBONE_BLOCKS} backbone blocks for adaptive gradual unfreeze')

    # Helper to strip possible 'module.' prefixes and ensure string keys
    def strip_module_prefix(d):
        new = {}
        try:
            items = d.items()
        except Exception:
            # if it's an iterator like named_parameters() dict-like, try to coerce
            items = list(d)
        for k, v in list(d.items()):
            if isinstance(k, bytes):
                try:
                    k = k.decode()
                except Exception:
                    k = str(k)
            if isinstance(k, str):
                nk = k.replace('module.', '')
            else:
                nk = k
            new[nk] = v
        return new

    # Helper: attempt to inject a HF transformers model state_dict into our model
    def inject_hf_weights(hf_model_id, target_model):
        try:
            from transformers import AutoModelForImageClassification
        except Exception as e:
            logger.error('Transformers not installed: please pip install transformers')
            raise
        logger.info('Downloading HF model %s', hf_model_id)
        hf = AutoModelForImageClassification.from_pretrained(hf_model_id)
        # Try several locations for state_dict to be robust across HF model wrappers
        hf_sd = None
        tried = []
        try:
            sd = hf.state_dict()
            hf_sd = sd
            tried.append('hf.state_dict')
        except Exception:
            pass
        if hf_sd is None:
            for attr in ('base_model', 'model', 'vision_model', 'efficientnet', 'backbone'):
                if hasattr(hf, attr):
                    try:
                        sd = getattr(hf, attr).state_dict()
                        hf_sd = sd
                        tried.append(f'hf.{attr}.state_dict')
                        break
                    except Exception:
                        continue
        if hf_sd is None:
            # as a last resort, try huggingface's from_pretrained returned object as mapping
            try:
                hf_sd = dict(hf.named_parameters())
                tried.append('hf.named_parameters')
            except Exception:
                pass
        if hf_sd is None:
            raise RuntimeError('Unable to extract state_dict from HF model; tried: ' + ','.join(tried))
        # strip module prefixes from hf keys
        hf_sd = strip_module_prefix(hf_sd)
        logger.info('Extracted HF state_dict using: %s', ','.join(tried))
        target_sd = target_model.state_dict()
        mapped = {}
        matched = 0
        # Simple heuristics: direct match, remove/add common prefixes, suffix match
        hf_keys = set(hf_sd.keys())
        used_hf_keys = set()
        for t_k in target_sd.keys():
            hkname = None
            if t_k in hf_sd:
                hkname = t_k
            else:
                # try removing 'backbone.' prefix
                if t_k.startswith('backbone.'):
                    cand = t_k.replace('backbone.', '')
                    if cand in hf_sd:
                        hkname = cand
                    else:
                        cand2 = 'features.' + cand
                        if cand2 in hf_sd:
                            hkname = cand2
                # try suffix match
                if hkname is None:
                    for hk in hf_keys:
                        if hk.endswith(t_k):
                            hkname = hk
                            break
            if hkname is not None:
                mapped[t_k] = hf_sd[hkname]
                used_hf_keys.add(hkname)
                matched += 1
                continue
        logger.info('HF inject: matched %d/%d target parameters (%.1f%%)', matched, len(target_sd), matched / max(1, len(target_sd)) * 100.0)
        if STRICT_INJECT:
            pct = matched / max(1, len(target_sd))
            if pct < 0.5:
                # attempt a shape-based fallback to rescue some matches
                logger.info('Strict inject: only %.1f%% matched, attempting shape-based fallback', pct * 100.0)
                # shape-based mapping: match by identical tensor shapes when possible
                for t_k, t_v in target_sd.items():
                    if t_k in mapped:
                        continue
                    t_shape = None
                    try:
                        t_shape = tuple(t_v.shape)
                    except Exception:
                        continue
                    for hk in hf_keys:
                        if hk in used_hf_keys:
                            continue
                        try:
                            h_v = hf_sd[hk]
                            if tuple(getattr(h_v, 'shape', ())) == t_shape:
                                mapped[t_k] = h_v
                                used_hf_keys.add(hk)
                                matched += 1
                                break
                        except Exception:
                            continue
                pct2 = matched / max(1, len(target_sd))
                logger.info('After shape fallback matched %d/%d (%.1f%%)', matched, len(target_sd), pct2 * 100.0)
                if pct2 < 0.5:
                    raise RuntimeError(f'Strict inject failed: only {matched}/{len(target_sd)} ({pct2:.2f}) keys matched from HF model')
        # load mapped keys into model (non-strict)
        target_model.load_state_dict(mapped, strict=False)
        return matched

    # If HF model requested, inject its weights first (unless from-scratch requested)
    # Option A: use HF model object directly (recommended when architectures differ)
    if HF_MODEL_ID and USE_HF_DIRECT and not FROM_SCRATCH:
        try:
            from transformers import AutoModelForImageClassification
            logger.info('Loading HF model object directly: %s', HF_MODEL_ID)
            hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
            # If HF model head has a different number of labels than our target, replace the head
            hf_num_labels = getattr(hf_model.config, 'num_labels', None)
            if hf_num_labels is not None and hf_num_labels != num_classes:
                logger.info('HF model has %s labels but target num_classes=%s; replacing classifier head', hf_num_labels, num_classes)
                # Try common classifier attributes (classifier, head, heads)
                replaced = False
                try:
                    if hasattr(hf_model, 'classifier') and hasattr(hf_model.classifier, 'in_features'):
                        in_f = hf_model.classifier.in_features
                        hf_model.classifier = nn.Linear(in_f, num_classes)
                        replaced = True
                    elif hasattr(hf_model, 'head') and hasattr(hf_model.head, 'in_features'):
                        in_f = hf_model.head.in_features
                        hf_model.head = nn.Linear(in_f, num_classes)
                        replaced = True
                    elif hasattr(hf_model, 'heads'):
                        # some models expose heads as a ModuleDict
                        # attempt a conservative replacement for common keys
                        for k in ('classifier', 'head', 'head0'):
                            if k in hf_model.heads:
                                in_f = hf_model.heads[k].in_features
                                hf_model.heads[k] = nn.Linear(in_f, num_classes)
                                replaced = True
                                break
                except Exception:
                    replaced = False
                if not replaced:
                    logger.warning('Could not auto-replace HF classifier head. You may need to adapt the HF model manually.')
            # Replace local model with the HF model for training/eval
            model = hf_model
            logger.info('Replaced local BirdClassifier with HF model object: %s', HF_MODEL_ID)
        except Exception as e:
            logger.exception('Failed to load HF model directly: %s', e)
            raise
    # Option B: inject HF weights into local model (existing heuristic mapping)
    elif HF_MODEL_ID and not FROM_SCRATCH:
        try:
            injected = inject_hf_weights(HF_MODEL_ID, model)
            logger.info('Injected HF model weights (%d keys matched)', injected)
        except Exception as e:
            logger.exception('HF injection failed: %s', e)
            raise

    # If a resume or default BEST_MODEL_PATH exists and not from-scratch, load checkpoint file
    state = None
    if not FROM_SCRATCH and (RESUME_CKPT or (HF_MODEL_ID is None)):
        ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
        # Robust checkpoint handling: look for common wrapper keys then fall back to assuming a raw state_dict
        if isinstance(ckpt, dict):
            for candidate in ('state_dict', 'model_state_dict', 'model_state', 'model'):
                if candidate in ckpt:
                    state = ckpt[candidate]
                    logger.info("Loaded checkpoint container key: '%s' from %s", candidate, BEST_MODEL_PATH)
                    break
            if state is None:
                try:
                    sample_vals = list(ckpt.values())[:5]
                    if all(hasattr(v, 'dtype') or hasattr(v, 'shape') for v in sample_vals):
                        state = ckpt
                    else:
                        for v in ckpt.values():
                            if isinstance(v, dict):
                                state = v
                                logger.info('Auto-selected nested dict from checkpoint as state_dict')
                                break
                except Exception:
                    state = ckpt
        else:
            state = ckpt

    # Checkpoint loading is handled above according to --from-scratch and --resume.
    # We intentionally avoid an unconditional load of BEST_MODEL_PATH here so that
    # providing --hf-model / --use-hf-direct won't be silently overridden by a
    # default checkpoint. If you want to resume a previous run, pass --resume <path>.

    if isinstance(state, dict):
        any_module_prefixed = any(isinstance(k, str) and k.startswith('module.') for k in state.keys())
        if any_module_prefixed:
            state = strip_module_prefix(state)

    # Heuristic: if the selected 'state' still looks like an outer checkpoint (has 'epoch' or 'optimizer_state_dict'), try to unwrap nested dicts
    def looks_like_state_dict_like(d):
        if not isinstance(d, dict):
            return False
        # If any key looks like model layer prefix, assume it's the correct state dict
        for k in d.keys():
            if isinstance(k, str) and (k.startswith('backbone') or k.startswith('features') or k.startswith('conv') or k.startswith('classifier') or k.startswith('layer')):
                return True
        # If values look like tensors (have dtype/shape), assume state dict
        sample_vals = list(d.values())[:8]
        tensor_like = 0
        for v in sample_vals:
            if hasattr(v, 'dtype') or hasattr(v, 'shape'):
                tensor_like += 1
        if tensor_like >= 1:
            return True
        return False

    if isinstance(state, dict) and not looks_like_state_dict_like(state):
        # try common nested keys
        for cand in ('state_dict', 'model_state_dict', 'model_state', 'model'):
            if cand in state and isinstance(state[cand], dict) and looks_like_state_dict_like(state[cand]):
                print(f"Unwrapping nested checkpoint key '{cand}' as model state_dict")
                state = state[cand]
                break
        else:
            # try to auto-find a nested dict that looks like a model state_dict
            for v in state.values():
                if isinstance(v, dict) and looks_like_state_dict_like(v):
                    print('Auto-unwrapped nested dict as model state_dict')
                    state = v
                    break

    # Validate that we ended up with a mapping-like state dict.
    # If state is None, allow it only when using HF initialization or when training from scratch.
    if state is None:
        if not FROM_SCRATCH and HF_MODEL_ID is None:
            raise RuntimeError(f'No valid state_dict found inside checkpoint: {BEST_MODEL_PATH}. Check the file contents.')
        # else: proceed without loading a checkpoint (HF-direct or from-scratch)
    else:
        # state is not None: ensure it's a dict-like state_dict, coerce when possible
        if not isinstance(state, dict):
            try:
                if isinstance(state, Mapping):
                    state = dict(state)
                else:
                    # last resort: attempt dict() conversion which may work for some custom mapping objects
                    state = dict(state)
                # sanitize keys: convert bytes keys to str
                try:
                    state = { (k.decode() if isinstance(k, bytes) else k): v for k, v in state.items() }
                except Exception:
                    pass
            except Exception:
                raise RuntimeError(f'Checkpoint loaded from {BEST_MODEL_PATH} did not contain a dict-like state_dict; got type {type(state)}')

    # Attempt strict load first; on failure fall back to non-strict and print diagnostics
    if state is not None:
        try:
            model.load_state_dict(state)
            logger.info('Checkpoint loaded with strict=True')
        except Exception as e:
            logger.warning('Strict load failed: %s', e)
            try:
                res = model.load_state_dict(state, strict=False)
                # res is an IncompatibleKeys namedtuple with missing_keys/unexpected_keys
                missing = getattr(res, 'missing_keys', None)
                unexpected = getattr(res, 'unexpected_keys', None)
                if missing is None and unexpected is None and isinstance(res, dict):
                    # Older torch may return dict-like info
                    missing = res.get('missing_keys', [])
                    unexpected = res.get('unexpected_keys', [])
                logger.info('Checkpoint loaded with strict=False; missing: %d, unexpected: %d', len(missing or []), len(unexpected or []))
                # Diagnostic: percentage of model keys matched
                model_keys = set(model.state_dict().keys())
                state_keys = set(state.keys())
                matched_keys = len(model_keys & state_keys)
                pct = matched_keys / max(1, len(model_keys)) * 100.0
                logger.info('Matched %d/%d model keys (%.1f%%)', matched_keys, len(model_keys), pct)
            except Exception as e2:
                logger.exception('Failed to load checkpoint from %s: %s', BEST_MODEL_PATH, e2)
                raise RuntimeError(f'Failed to load checkpoint from {BEST_MODEL_PATH}: {e2}')
    model.to(device)

    # Freeze backbone initially. Handle both local BirdClassifier (with names containing 'backbone')
    # and HF model objects (whose head often contains 'classifier' or 'head').
    for name, p in model.named_parameters():
        # If it's the project's BirdClassifier, backbone params contain 'backbone'
        if 'backbone' in name:
            p.requires_grad = False
        # For HF models loaded directly, treat common head names as trainable and others as backbone
        elif USE_HF_DIRECT and HF_MODEL_ID:
            if any(k in name for k in ('classifier', 'head', 'heads')):
                p.requires_grad = True
            else:
                p.requires_grad = False
        else:
            p.requires_grad = True

    # Freeze BatchNorm running stats during head-warmup to avoid noisy estimates on small batches
    backbone = getattr(model, 'backbone', None) or getattr(model, 'features', None)
    if backbone is not None:
        freeze_batchnorm_stats(backbone)

    # Loss and optimizer (class weights)
    if USE_CLASS_WEIGHTS:
        # compute class weights from train frequencies (inverse-frequency normalized)
        print('Computing class weights from training set frequencies...')
        computed_weights = calculate_class_weights(train_loader, num_classes, device='cpu')
        # move to device
        class_weights = computed_weights.to(device)
    else:
        # dynamic based on deficit from avg (legacy targeted boosting)
        class_weights = torch.ones(num_classes, device=device)
        for c in range(num_classes):
            if c in per_class_acc:
                deficit = max(0.0, avg_acc - per_class_acc[c])
                # normalize deficit by avg_acc and scale by BOOST_FACTOR
                mult = 1.0 + (deficit / max(1e-6, avg_acc)) * BOOST_FACTOR
                class_weights[c] = min(MAX_BOOST, float(mult))
            elif c in targets:
                class_weights[c] = min(MAX_BOOST, float(BOOST_FACTOR))

    # Optionally soften aggressive augmentations for fine-tuning
    if SOFTEN_AUG:
        # lower cutmix probability and set a gentle mixup if not specified
        try:
            if float(globals().get('CUTMIX_PROB', args.cutmix_prob)) > 0.5:
                globals()['CUTMIX_PROB'] = 0.2
        except Exception:
            globals()['CUTMIX_PROB'] = 0.2
        try:
            if float(globals().get('MIXUP_ALPHA', args.mixup_alpha)) <= 0.0:
                globals()['MIXUP_ALPHA'] = 0.1
        except Exception:
            globals()['MIXUP_ALPHA'] = 0.1

    if USE_FOCAL:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Prepare parameter groups: if no backbone params are trainable yet, omit that group
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': LR_BACKBONE})
    if head_params:
        param_groups.append({'params': head_params, 'lr': LR_HEAD})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    print(f'Out dir: {OUT_DIR}, Resuming from: {BEST_MODEL_PATH}')
    print('Early stopping enabled:', ENABLE_EARLY_STOP)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val = -1.0
    best_avg_class = -1.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'per_class_acc': []}
    # optional test metrics (filled only if ALLOW_TEST_SELECTION or when eval_split=='test')
    history['test_loss'] = []
    history['test_acc'] = []
    history['test_per_class_acc'] = []
    # record the primary evaluation split metric (val or test) per epoch
    history['primary_acc'] = []

    # bookkeeping for optional test-based selection
    best_test_top1 = -1.0
    best_test_avgclass = -1.0
    if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION:
        print('WARNING: Test-based checkpoint selection is ENABLED. This may introduce evaluation leakage. Proceed only if you understand the implications.')

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # always evaluate val for diagnostics
        val_loss, val_acc, val_per_class = evaluate(model, val_loader, criterion, device, num_classes)

        # decide whether to evaluate test this epoch (either explicit opt-in or user selected eval_split=test)
        do_test_eval = ('ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION) or (EVAL_SPLIT == 'test')
        if do_test_eval:
            test_loss, test_acc, test_per_class = evaluate(model, test_loader, criterion, device, num_classes)
        else:
            test_loss, test_acc, test_per_class = (None, None, None)

        # choose primary metrics based on requested eval split
        if EVAL_SPLIT == 'val':
            primary_loss, primary_acc, primary_per_class = val_loss, val_acc, val_per_class
        else:
            primary_loss, primary_acc, primary_per_class = test_loss, test_acc, test_per_class

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['per_class_acc'].append(val_per_class)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_per_class_acc'].append(test_per_class)
        history['primary_acc'].append(primary_acc)

        # compute avg per-class accuracy for the chosen evaluation split
        try:
            primary_avg_per_class = sum(primary_per_class) / len(primary_per_class) if primary_per_class else 0.0
        except Exception:
            primary_avg_per_class = 0.0

        # pretty print epoch summary including both val and primary split (if different)
        val_loss_s = f"{val_loss:.4f}" if val_loss is not None else 'nan'
        val_acc_s = f"{val_acc:.2f}%" if val_acc is not None else 'nan'
        prim_loss_s = f"{primary_loss:.4f}" if primary_loss is not None else 'nan'
        prim_acc_s = f"{primary_acc:.2f}%" if primary_acc is not None else 'nan'
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} train_acc: {train_acc:.2f}% | val_loss: {val_loss_s} val_acc: {val_acc_s} | {EVAL_SPLIT}_loss: {prim_loss_s} {EVAL_SPLIT}_acc: {prim_acc_s} | avg_class_acc({EVAL_SPLIT}): {primary_avg_per_class:.2f}% ({time.time()-t0:.1f}s)")

        # Save best by primary top-1 (val or test depending on --eval-split)
        if primary_acc is not None and primary_acc > best_val + 1e-4:
            best_val = primary_acc
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, f'{EVAL_SPLIT}_acc': primary_acc}, OUT_CKPT)
            print(f' -> New best saved ({EVAL_SPLIT}_acc {primary_acc:.2f}%)')

        # If enabled, save best by test top-1 (explicit opt-in)
        if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION and test_acc is not None:
            if test_acc > best_test_top1 + 1e-4:
                best_test_top1 = test_acc
                tk = os.path.join(OUT_DIR, 'best_model_test_top1.pth')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'test_acc': test_acc}, tk)
                print(' -> New test-top1 best saved (test_acc {:.2f}%) -> {}'.format(test_acc, tk))

        # Save best by average per-class accuracy on the chosen split
        if primary_avg_per_class > best_avg_class + 1e-4:
            best_avg_class = primary_avg_per_class
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, f'{EVAL_SPLIT}_avg_class': primary_avg_per_class}, OUT_CKPT.replace('.pth', f'_{EVAL_SPLIT}_avgclass.pth'))
            print(f' -> New best saved by avg-class-acc ({EVAL_SPLIT}) ({primary_avg_per_class:.2f}%)')

        # If enabled, save best by test avg-class (explicit opt-in)
        if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION and test_per_class is not None:
            test_avg_class = sum(test_per_class) / len(test_per_class)
            if test_avg_class > best_test_avgclass + 1e-4:
                best_test_avgclass = test_avg_class
                tk2 = os.path.join(OUT_DIR, 'best_model_test_avgclass.pth')
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'test_avg_class': test_avg_class}, tk2)
                print(' -> New test-avgclass best saved ({:.2f}%) -> {}'.format(test_avg_class, tk2))

    # Gradual unfreeze: start unfreezing parts of backbone at UNFREEZE_AT, full at UNFREEZE_AT+1
        # Backbone unfreezing schedule:
        # - epochs 1..FREEZE_EPOCHS: backbone fully frozen
        # - next GRADUAL_UNFREEZE_EPOCHS epochs: progressively unfreeze more blocks
        # - after that: full unfreeze and rebuild optimizer
        if epoch > FREEZE_EPOCHS and epoch <= FREEZE_EPOCHS + GRADUAL_UNFREEZE_EPOCHS:
            # how many steps into the gradual unfreeze we are (1..GRADUAL_UNFREEZE_EPOCHS)
            step = epoch - FREEZE_EPOCHS
            # adaptively compute how many of the backbone blocks to unfreeze at this step
            n_unfreeze = math.ceil(TOTAL_BACKBONE_BLOCKS * (step / float(GRADUAL_UNFREEZE_EPOCHS)))
            n_unfreeze = max(1, min(TOTAL_BACKBONE_BLOCKS, n_unfreeze))
            print(f'Gradual unfreeze step {step}/{GRADUAL_UNFREEZE_EPOCHS}: unfreezing last {n_unfreeze}/{TOTAL_BACKBONE_BLOCKS} backbone blocks...')
            unfrozen = gradual_unfreeze(model, backbone_attr='backbone', block_name_pattern=r'_blocks\.\d+', unfreeze_last_n_blocks=n_unfreeze)
            if unfrozen:
                print('Unfrozen params (this step):', len(unfrozen))
            else:
                # Fallback: try a name-based heuristic to unfreeze more selective layers
                for name, p in model.named_parameters():
                    if 'backbone' in name and ('layer4' in name or 'block5' in name or 'conv_head' in name):
                        p.requires_grad = True
        # Full unfreeze after the gradual schedule completes
        if epoch == FREEZE_EPOCHS + GRADUAL_UNFREEZE_EPOCHS + 1:
            print('Unfreezing full backbone for deeper fine-tuning...')
            for name, p in model.named_parameters():
                p.requires_grad = True
            # Rebuild optimizer with differential LR
            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': BACKBONE_LR_UNFREEZE},
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad], 'lr': LR_HEAD}
            ], weight_decay=WEIGHT_DECAY)
            # Re-enable BatchNorm updates once backbone is mostly unfrozen
            backbone = getattr(model, 'backbone', None) or getattr(model, 'features', None)
            if backbone is not None:
                unfreeze_batchnorm_stats(backbone)

        # Step scheduler based on the chosen primary avg per-class accuracy
        # step LR scheduler on the primary avg-per-class metric
        scheduler.step(primary_avg_per_class)
        # Save per-epoch checkpoint (keeps a record for later averaging)
        os.makedirs(OUT_DIR, exist_ok=True)
        epoch_ckpt = os.path.join(OUT_DIR, f'ckpt_epoch_{epoch}.pth')
        ckpt_to_save = {'state_dict': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc, f'{EVAL_SPLIT}_acc': primary_acc, 'avg_class_acc': primary_avg_per_class}
        # include test metrics in per-epoch checkpoint when available
        if 'ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION and test_per_class is not None:
            try:
                ckpt_to_save['test_acc'] = test_acc
                ckpt_to_save['test_per_class'] = test_per_class
            except Exception:
                pass
        torch.save(ckpt_to_save, epoch_ckpt)

        # Early stopping based on avg per-class accuracy (optional)
        if ENABLE_EARLY_STOP:
            if primary_avg_per_class <= best_avg_class + 1e-4:
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

    # Optional: auto-select best epoch checkpoint by val_acc and copy to OUT_CKPT
    if AUTO_SELECT_BEST:
        best_path = None
        best_metric = -1.0
        if SELECT_BASED_ON == 'test_avg_class' and not ('ALLOW_TEST_SELECTION' in globals() and ALLOW_TEST_SELECTION):
            print('Warning: --select-based-on test_avg_class selected but --allow-test-selection not enabled; falling back to val_avg_class')
            sel = 'val_avg_class'
        else:
            sel = SELECT_BASED_ON

        for _, p in all_ckpts:
            try:
                ck = torch.load(p, map_location='cpu')
                if sel == 'test_avg_class':
                    tpc = ck.get('test_per_class', None)
                    if tpc is None:
                        continue
                    v = (sum(tpc) / len(tpc)) if tpc else None
                else:
                    v = ck.get('avg_class_acc', None)
                if v is not None and v > best_metric:
                    best_metric = v
                    best_path = p
            except Exception:
                continue
        if best_path:
            shutil.copy(best_path, OUT_CKPT)
            print(f'Auto-selected best epoch checkpoint {best_path} ({sel}={best_metric:.2f}) -> {OUT_CKPT}')
        else:
            print('AUTO_SELECT_BEST enabled but no valid epoch checkpoints found to select')

    print('Fine-tuning complete. Best val acc: {:.2f}%'.format(best_val))
    print('Checkpoint saved to:', OUT_CKPT)

