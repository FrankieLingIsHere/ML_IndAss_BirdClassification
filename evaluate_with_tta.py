import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
import numpy as np
from models import BirdClassifier
from data_loader import BirdDataset
import argparse

MODEL_PATH = 'results_stage2_accelerated/best_model_finetuned.pth'
TEST_DIR = 'data/Test'
TEST_TXT = 'data/test.txt'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define TTA transforms (base is center crop no-aug)
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
    transforms.RandomResizedCrop(384, scale=(0.9,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

TRANSFORMS = [base, flip, crop]


def load_model(model_path, num_classes=200):
    ckpt = torch.load(model_path, map_location='cpu')
    # extract state
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    # strip module.
    new_state = {}
    for k,v in state.items():
        nk = k.replace('module.','') if k.startswith('module.') else k
        new_state[nk]=v
    model = BirdClassifier(num_classes=num_classes, architecture='efficientnet_b3', pretrained=False, dropout_rate=0.3)
    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_tta(model, tta_transforms=TRANSFORMS):
    # Iterate directly over sample paths to avoid DataLoader collate issues
    test_dataset = BirdDataset(TEST_DIR, TEST_TXT, transform=None)
    samples = test_dataset.samples
    total = 0
    correct = 0
    per_class_total = [0] * 200
    per_class_correct = [0] * 200

    import os

    from PIL import Image
    with torch.no_grad():
        for path, label in samples:
            pil = Image.open(path).convert('RGB')
            logits = None
            for tr in tta_transforms:
                inp = tr(pil).unsqueeze(0).to(DEVICE)
                out = model(inp)
                if logits is None:
                    logits = out
                else:
                    logits = logits + out
            preds = logits.argmax(dim=1).cpu().numpy()
            lab = int(label)
            total += 1
            if preds[0] == lab:
                correct += 1
                per_class_correct[lab] += 1
            per_class_total[lab] += 1
            if total % 50 == 0:
                print('Processed {} images, current acc {:.2f}%'.format(total, correct / total * 100))

    top1 = correct/total*100
    class_accs = []
    for i in range(200):
        class_accs.append((per_class_correct[i]/per_class_total[i]*100) if per_class_total[i]>0 else 0.0)
    avg_class = sum(class_accs)/200
    return top1, avg_class, class_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint to evaluate')
    parser.add_argument('--out-dir', type=str, default='./results/eval_tta', help='Directory to save TTA results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = load_model(args.model)
    top1, avg_class, class_accs = evaluate_tta(model)
    print('TTA Top-1: {:.2f}%  Avg per-class: {:.2f}%'.format(top1, avg_class))
    out_path = os.path.join(args.out_dir, 'tta_results.json')
    with open(out_path, 'w') as f:
        json.dump({'top1': float(top1), 'avg_class': float(avg_class), 'class_accs': class_accs, 'model': args.model}, f, indent=2)
    print('Saved to', out_path)
