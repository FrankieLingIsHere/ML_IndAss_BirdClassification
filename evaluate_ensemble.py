import argparse
import os
import json
from PIL import Image
import torch
from torchvision import transforms
from models import BirdClassifier
from data_loader import BirdDataset

# Default paths
BASE_RESULTS_DIR = 'results_stage2_accelerated'
DEFAULT_MODELS = [os.path.join(BASE_RESULTS_DIR, 'best_model_finetuned.pth'), os.path.join(BASE_RESULTS_DIR, 'best_model.pth')]
TEST_DIR = 'data/Test'
TEST_TXT = 'data/test.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 200

# TTA transforms
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


def ensemble_evaluate(model_paths, use_tta=True, tta_transforms=DEFAULT_TTA):
    models = []
    for p in model_paths:
        if not os.path.exists(p):
            print('Warning: model not found, skipping:', p)
            continue
        print('Loading model:', p)
        models.append(load_model(p))
    if not models:
        raise RuntimeError('No valid models to ensemble')

    dataset = BirdDataset(TEST_DIR, TEST_TXT, transform=None)
    samples = dataset.samples

    total = 0
    correct = 0
    per_class_total = [0] * NUM_CLASSES
    per_class_correct = [0] * NUM_CLASSES

    with torch.no_grad():
        for path, label in samples:
            pil = Image.open(path).convert('RGB')
            label = int(label)
            # accumulate logits across models and TTA using explicit accumulators
            ensemble_accum = torch.zeros((1, NUM_CLASSES), device=DEVICE)
            for model in models:
                # per-model accumulator for TTA
                if use_tta and len(tta_transforms) > 0:
                    n_tta = len(tta_transforms)
                    tta_accum = torch.zeros((1, NUM_CLASSES), device=DEVICE)
                    for tr in tta_transforms:
                        inp = tr(pil).unsqueeze(0).to(DEVICE)
                        out = model(inp)
                        tta_accum += out
                    # average over TTA transforms
                    model_logits = tta_accum / float(n_tta)
                else:
                    inp = base(pil).unsqueeze(0).to(DEVICE)
                    model_logits = model(inp)

                ensemble_accum += model_logits

            # average across models (models guaranteed non-empty)
            ensemble_logits = ensemble_accum / float(len(models))
            pred = int(ensemble_logits.argmax(dim=1).cpu().numpy()[0])
            total += 1
            per_class_total[label] += 1
            if pred == label:
                correct += 1
                per_class_correct[label] += 1
            if total % 100 == 0:
                print('Processed {}, current acc {:.2f}%'.format(total, correct/total*100))

    top1 = correct / total * 100.0
    class_accs = [(per_class_correct[i] / per_class_total[i] * 100.0) if per_class_total[i] > 0 else 0.0 for i in range(NUM_CLASSES)]
    avg_class = sum(class_accs) / NUM_CLASSES
    return top1, avg_class, class_accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Model checkpoints to ensemble')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    args = parser.parse_args()

    model_paths = args.models
    use_tta = not args.no_tta

    print('Device:', DEVICE)
    print('Ensembling models:', model_paths)
    top1, avg_class, class_accs = ensemble_evaluate(model_paths, use_tta=use_tta)
    print('\nENSEMBLE RESULTS:')
    print('Top-1: {:.2f}%  Avg-class: {:.2f}%'.format(top1, avg_class))
    out = {'top1': float(top1), 'avg_class': float(avg_class), 'class_accs': class_accs, 'models': model_paths, 'tta': use_tta}
    with open('ensemble_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved ensemble_results.json')
