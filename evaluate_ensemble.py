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


def ensemble_evaluate(model_paths, use_tta=True, tta_transforms=DEFAULT_TTA, weights=None):
    models = []
    # If weights provided, they should align with model_paths. We'll keep only weights for loaded models.
    provided_weights = list(weights) if weights is not None else None
    used_weights = [] if provided_weights is not None else None
    for idx, p in enumerate(model_paths):
        if not os.path.exists(p):
            print('Warning: model not found, skipping:', p)
            continue
        print('Loading model:', p)
        models.append(load_model(p))
        if used_weights is not None:
            # protect against weights shorter than model_paths
            if provided_weights is not None and idx < len(provided_weights):
                used_weights.append(float(provided_weights[idx]))
            else:
                print('Warning: not enough weights provided for all models; remaining models will get equal weight.')
                used_weights.append(None)
    if not models:
        raise RuntimeError('No valid models to ensemble')

    # Normalize/resolve weights to a tensor aligned with loaded models
    if used_weights is None:
        weights_tensor = torch.tensor([1.0 / len(models)] * len(models), device=DEVICE)
    else:
        # Replace any None with equal share of remaining weight
        resolved = []
        # If any None entries exist, set them temporarily to 1.0; we'll normalize below
        for w in used_weights:
            resolved.append(1.0 if w is None else float(w))
        wsum = sum(resolved)
        if wsum <= 0:
            print('Warning: provided weights sum to zero or negative, falling back to equal weights')
            weights_tensor = torch.tensor([1.0 / len(models)] * len(models), device=DEVICE)
        else:
            normalized = [float(w) / wsum for w in resolved]
            weights_tensor = torch.tensor(normalized, device=DEVICE)

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
            for i, model in enumerate(models):
                model_weight = float(weights_tensor[i]) if weights_tensor is not None else (1.0 / float(len(models)))
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

                # accumulate weighted logits
                ensemble_accum += model_logits * model_weight

            # ensemble_accum already holds weighted logits normalized to sum=1, so use directly
            ensemble_logits = ensemble_accum
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
    # return weights as a list as well for JSON output
    return top1, avg_class, class_accs, weights_tensor.cpu().tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Model checkpoints to ensemble')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    parser.add_argument('--weights', nargs='+', type=float, help='Optional weights for each model (aligned with --models)')
    parser.add_argument('--out-dir', default=os.path.join('results', 'ensemble'), help='Directory to write the ensemble_results.json')
    args = parser.parse_args()

    model_paths = args.models
    use_tta = not args.no_tta
    weights_arg = args.weights

    print('Device:', DEVICE)
    print('Ensembling models:', model_paths)
    top1, avg_class, class_accs, used_weights = ensemble_evaluate(model_paths, use_tta=use_tta, weights=weights_arg)
    print('\nENSEMBLE RESULTS:')
    print('Top-1: {:.2f}%  Avg-class: {:.2f}%'.format(top1, avg_class))
    # include the (normalized) weights used in the JSON output when available
    out = {'top1': float(top1), 'avg_class': float(avg_class), 'class_accs': class_accs, 'models': model_paths, 'tta': use_tta}
    out['weights'] = used_weights
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'ensemble_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved {out_path}')
