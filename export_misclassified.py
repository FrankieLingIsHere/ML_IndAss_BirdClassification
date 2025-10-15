"""
Runs a model or ensemble on the Test set, writes a CSV of predictions and misclassified samples,
and copies a small sample of misclassified images into `misclassified/<class_id>/` for manual inspection.

Usage:
    python export_misclassified.py --models path1.pth path2.pth --no-tta --copy 5

Defaults: uses the same default models as `evaluate_ensemble.py` and TTA enabled.
"""
import argparse
import os
import csv
import json
import shutil
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms
from models import BirdClassifier
from data_loader import BirdDataset

BASE_RESULTS_DIR = 'results_stage2_accelerated'
DEFAULT_MODELS = [os.path.join(BASE_RESULTS_DIR, 'best_model_finetuned.pth'), os.path.join(BASE_RESULTS_DIR, 'best_model.pth')]
TEST_DIR = 'data/Test'
TEST_TXT = 'data/test.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 200

# reuse transforms from evaluate_ensemble.py
base = transforms.Compose([
    transforms.Resize((int(384*1.05), int(384*1.05))),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
flip = transforms.Compose([
    transforms.Resize((int(384*1.05), int(384*1.05))),
    transforms.CenterCrop(384),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
crop = transforms.Compose([
    transforms.Resize((int(384*1.1), int(384*1.1))),
    transforms.RandomResizedCrop(384, scale=(0.95,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
DEFAULT_TTA = [base, flip, crop]


def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        nk = k.replace('module.', '') if k.startswith('module.') else k
        new_state[nk] = v
    model = BirdClassifier(num_classes=NUM_CLASSES, architecture='efficientnet_b3', pretrained=False, dropout_rate=0.3)
    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def predict_on_sample(pil, models, use_tta=True, tta_transforms=DEFAULT_TTA):
    ensemble_accum = torch.zeros((1, NUM_CLASSES), device=DEVICE)
    for model in models:
        if use_tta and len(tta_transforms) > 0:
            n_tta = len(tta_transforms)
            tta_accum = torch.zeros((1, NUM_CLASSES), device=DEVICE)
            for tr in tta_transforms:
                inp = tr(pil).unsqueeze(0).to(DEVICE)
                out = model(inp)
                tta_accum += out
            model_logits = tta_accum / float(n_tta)
        else:
            inp = base(pil).unsqueeze(0).to(DEVICE)
            model_logits = model(inp)
        ensemble_accum += model_logits
    ensemble_logits = ensemble_accum / float(len(models))
    pred = int(ensemble_logits.argmax(dim=1).cpu().numpy()[0])
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Model checkpoints to ensemble')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    parser.add_argument('--copy', type=int, default=0, help='Copy up to N misclassified examples per class into misclassified/<class_id>/')
    parser.add_argument('--out-dir', type=str, default=os.path.join('results', 'misclassified'), help='Directory to write misclassified.csv and copied images')
    args = parser.parse_args()

    models = []
    for p in args.models:
        if os.path.exists(p):
            print('Loading', p)
            models.append(load_model(p))
        else:
            print('Warning: model not found, skipping', p)
    if not models:
        raise RuntimeError('No models loaded')

    dataset = BirdDataset(TEST_DIR, TEST_TXT, transform=None)
    samples = dataset.samples

    out_csv = os.path.join(args.out_dir, 'misclassified.csv')
    rows = []
    per_class_mis = defaultdict(list)

    with torch.no_grad():
        total = 0
        correct = 0
        for path, label in samples:
            pil = Image.open(path).convert('RGB')
            label = int(label)
            pred = predict_on_sample(pil, models, use_tta=not args.no_tta)
            rows.append((path, label, pred))
            total += 1
            if pred != label:
                per_class_mis[label].append(path)
            else:
                correct += 1
            if total % 200 == 0:
                print('Processed', total)

    # write CSV
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'pred'])
        writer.writerows(rows)
    print('Wrote', out_csv)

    # show worst classes
    class_errors = []
    for cid in range(NUM_CLASSES):
        total_c = sum(1 for _, l, _ in rows if l == cid)
        mis_c = len(per_class_mis.get(cid, []))
        err_rate = mis_c / total_c * 100.0 if total_c > 0 else 0.0
        class_errors.append((cid, total_c, mis_c, err_rate))
    class_errors.sort(key=lambda x: x[3], reverse=True)

    print('\nWorst classes by error rate (top 10):')
    for cid, total_c, mis_c, err in class_errors[:10]:
        print(f'class {cid}: {mis_c}/{total_c} misclassified ({err:.1f}%)')

    # copy examples
    if args.copy and args.copy > 0:
        for cid, paths in per_class_mis.items():
            if not paths:
                continue
            dest = os.path.join(args.out_dir, str(cid))
            os.makedirs(dest, exist_ok=True)
            for i, p in enumerate(paths[:args.copy]):
                # copy to dest
                shutil.copy(p, os.path.join(dest, os.path.basename(p)))
        print('Copied up to', args.copy, 'examples per class into', args.out_dir)
