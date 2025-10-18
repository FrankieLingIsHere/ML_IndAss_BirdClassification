"""
Final Model Evaluation Script for Bird Classification
Calculates Top-1 accuracy and Average accuracy per class as required.
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import csv
from models import BirdClassifier
from data_loader import BirdDataset, get_data_transforms, create_data_loaders
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import argparse
import glob

def load_model(model_path, num_classes=200, device=torch.device('cpu')):
    """Load the trained model.

    This loader handles different checkpoint formats:
    - raw state_dict saved with torch.save(model.state_dict())
    - wrapped checkpoints like {'state_dict': ..., 'epoch': ..., 'val_acc': ...}
    - DataParallel checkpoints with 'module.' prefixes
    It will attempt strict loading first and fall back to prefix-cleaning and non-strict loading if needed.
    """
    print("Loading model from: {}".format(model_path))

    # Create model with EfficientNet-B4 architecture (match training)
    model = BirdClassifier(
        num_classes=num_classes,
        architecture='efficientnet_b4',
        pretrained=False,
        dropout_rate=0.3
    )

    # Load checkpoint and extract state_dict if wrapped
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        # Common wrapper keys
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state' in ckpt:
            state = ckpt['model_state']
        else:
            # Might already be a state_dict but packaged with other keys
            # Try to detect by finding tensors in values
            maybe_state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            if maybe_state:
                state = maybe_state
            else:
                # Last resort: assume ckpt itself is the state dict
                state = ckpt
    else:
        state = ckpt

    # If keys are prefixed with 'module.', strip this (DataParallel compatibility)
    def strip_module_prefix(state_dict):
        new_state = {}
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            new_state[new_k] = v
        return new_state

    try:
        # Try strict loading first
        model.load_state_dict(state)
    except RuntimeError as e:
        # Attempt to clean prefixes and try again
        try:
            cleaned = strip_module_prefix(state)
            model.load_state_dict(cleaned, strict=False)
            print('Warning: loaded state_dict with strict=False after cleaning module prefixes.')
        except Exception:
            # Final fallback: try loading with non-strict as-is
            model.load_state_dict(state, strict=False)
            print('Warning: loaded state_dict with strict=False (fallback).')

    model.to(device)
    model.eval()
    return model

def _find_image_in_dir(image_name: str, image_dir: str):
    """Find the full path to an image within a directory (tries extensions and subfolders)."""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    # If already has extension
    if any(image_name.lower().endswith(ext) for ext in extensions):
        candidates = [os.path.join(image_dir, image_name)]
        for root, dirs, files in os.walk(image_dir):
            if image_name in files:
                candidates.append(os.path.join(root, image_name))
    else:
        candidates = []
        for ext in extensions:
            fname = image_name + ext
            candidates.append(os.path.join(image_dir, fname))
            for root, dirs, files in os.walk(image_dir):
                if fname in files:
                    candidates.append(os.path.join(root, fname))

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


class SimpleTestDataset(Dataset):
    """Lightweight dataset built from explicit (image_path, label_idx) pairs."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), 0)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), 0)
            if self.transform:
                image = self.transform(image)
            return image, label


def create_test_loader_with_train_mapping(train_dir: str, train_txt: str, test_dir: str, test_txt: str,
                                          batch_size: int = 32, image_size: int = 384, num_workers: int = 0,
                                          augmentation_level: str = 'advanced'):
    """Create a test loader that enforces the training class mapping.

    This reads the training class names via create_data_loaders and then parses the test
    annotations to produce labels that align with the training class->index mapping.
    """
    # Obtain class names from training dataset (we only need the mapping)
    _, _, _, num_classes, class_names = create_data_loaders(
        train_dir, train_txt, test_dir, test_txt, batch_size=1, image_size=image_size, num_workers=num_workers,
        validation_split=0.1, augmentation_level=augmentation_level
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Validation/test transform (no augmentation)
    test_transform = get_data_transforms(image_size, is_training=False)

    # Parse test annotation file and build samples list using training mapping
    samples = []
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"Test annotation file not found: {test_txt}")

    with open(test_txt, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            image_name = parts[0]
            label_token = ' '.join(parts[1:]) if len(parts) > 1 else ''

            # Determine label index
            label_idx = None
            # If label token is numeric, use as index (but ensure it's in range)
            try:
                cand = int(label_token)
                if 0 <= cand < len(class_names):
                    label_idx = cand
            except Exception:
                pass

            # If label token is a species name, map using training mapping
            if label_idx is None and label_token:
                if label_token in class_to_idx:
                    label_idx = class_to_idx[label_token]

            # Fallback: try to infer species name from filename (prefix before digits)
            if label_idx is None:
                name_parts = image_name.replace('.jpg', '').split('_')
                species_name = ''
                for i, part in enumerate(name_parts):
                    if part.isdigit():
                        species_name = '_'.join(name_parts[:i])
                        break
                if species_name and species_name in class_to_idx:
                    label_idx = class_to_idx[species_name]

            if label_idx is None:
                # Last resort: set to -1 and warn
                print(f"Warning: could not map test label '{label_token}' for image {image_name}; assigning -1")
                label_idx = -1

            img_path = _find_image_in_dir(image_name, test_dir)
            if img_path is None:
                print(f"Warning: could not find image {image_name} in {test_dir}")
                img_path = ''

            samples.append((img_path, label_idx))

    dataset = SimpleTestDataset(samples, transform=test_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return loader, class_names

def evaluate_model(model, test_loader, device, return_preds=False):
    """
    Evaluate model and calculate required metrics:
    1. Top-1 accuracy = (1/N) * Σ(argmax(y) == ground_truth)
    2. Average accuracy per class = (1/C) * Σ(T_i)
    """
    model.eval()
    
    # Storage for predictions and targets
    all_predictions = []
    all_targets = []
    
    # Per-class accuracy tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    total_correct = 0
    total_samples = 0
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions (argmax for Top-1)
            _, predicted = torch.max(outputs, 1)
            
            # Store all predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Calculate total correct for Top-1 accuracy
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update per-class statistics
            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
            
            # Progress indicator
            if (batch_idx + 1) % 20 == 0:
                current_acc = (total_correct / total_samples) * 100
                print("Processed {} batches, Current accuracy: {:.2f}%".format(batch_idx + 1, current_acc))
    
    # Calculate Top-1 Accuracy
    top1_accuracy = (total_correct / total_samples) * 100
    
    # Calculate Average Accuracy per Class
    class_accuracies = []
    per_class_details = {}
    
    # Get all unique classes that appeared in the test set
    all_classes = set(all_targets)
    
    for class_id in sorted(all_classes):
        if class_total[class_id] > 0:
            accuracy = (class_correct[class_id] / class_total[class_id]) * 100
            class_accuracies.append(accuracy)
            per_class_details[class_id] = {
                'accuracy': accuracy,
                'correct': class_correct[class_id],
                'total': class_total[class_id]
            }
        else:
            class_accuracies.append(0.0)
            per_class_details[class_id] = {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0
            }
    
    # Average accuracy per class
    average_accuracy_per_class = np.mean(class_accuracies)
    
    results = {
        'top1_accuracy': top1_accuracy,
        'average_accuracy_per_class': average_accuracy_per_class,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'num_classes_tested': len(all_classes),
        'per_class_details': per_class_details,
        'class_accuracies': class_accuracies
    }
    
    if return_preds:
        return results, all_predictions, all_targets
    return results

def load_class_names(filepath='class_names.json'):
    """Load class names if available"""
    try:
        with open(filepath, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        print("Class names file {} not found. Using class indices.".format(filepath))
        return None

def print_results(results, class_names=None):
    """Print comprehensive evaluation results"""
    print("\n" + "="*80)
    print("BIRD SPECIES CLASSIFICATION - FINAL EVALUATION RESULTS")
    print("="*80)
    
    # Primary metrics (as required)
    print("\nPRIMARY EVALUATION METRICS:")
    print("="*40)
    print("Top-1 Accuracy: {:.2f}%".format(results['top1_accuracy']))
    print("Average Accuracy per Class: {:.2f}%".format(results['average_accuracy_per_class']))
    
    # Additional details
    print("\nDETAILED STATISTICS:")
    print("="*40)
    print("Total test samples: {}".format(results['total_samples']))
    print("Correctly classified: {}".format(results['total_correct']))
    print("Number of classes in test set: {}".format(results['num_classes_tested']))
    
    # Per-class statistics
    class_accs = [details['accuracy'] for details in results['per_class_details'].values()]
    if class_accs:
        print("Best class accuracy: {:.2f}%".format(max(class_accs)))
        print("Worst class accuracy: {:.2f}%".format(min(class_accs)))
        print("Standard deviation: {:.2f}%".format(np.std(class_accs)))

        # Show top 5 best and worst performing classes
        sorted_classes = sorted(
            results['per_class_details'].items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )

        print("\nTOP 5 BEST PERFORMING CLASSES:")
        print("="*40)
        for i, (class_id, details) in enumerate(sorted_classes[:5]):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else "Class_{}".format(class_id)
            print("{}. {}: {:.2f}% ({}/{})".format(i+1, class_name, details['accuracy'], details['correct'], details['total']))

        print("\nTOP 5 WORST PERFORMING CLASSES:")
        print("="*40)
        for i, (class_id, details) in enumerate(sorted_classes[-5:]):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else "Class_{}".format(class_id)
            print("{}. {}: {:.2f}% ({}/{})".format(i+1, class_name, details['accuracy'], details['correct'], details['total']))

def save_results(results, filepath='evaluation_results.json'):
    """Save results to JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    results_json = {
        'top1_accuracy': float(results['top1_accuracy']),
        'average_accuracy_per_class': float(results['average_accuracy_per_class']),
        'total_samples': int(results['total_samples']),
        'total_correct': int(results['total_correct']),
        'num_classes_tested': int(results['num_classes_tested']),
        'per_class_details': {
            str(k): {
                'accuracy': float(v['accuracy']),
                'correct': int(v['correct']),
                'total': int(v['total'])
            } for k, v in results['per_class_details'].items()
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to: {}".format(filepath))

def main():
    """Main evaluation function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    
    # CLI: allow specifying a model path; otherwise auto-select
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint (.pth). If omitted, prefers best_model_finetuned.pth then best_model.pth in results_stage2_accelerated')
    parser.add_argument('--out-dir', type=str, default=os.path.join('results', 'eval_single'), help='Directory to write final_evaluation_results.json')
    parser.add_argument('--split', type=str, default='test', choices=['test','val'], help="Which split to evaluate: 'test' uses data/Test, 'val' uses the validation split from training via create_data_loaders")
    parser.add_argument('--raw-test-mapping', action='store_true', help='If set, use the test file mapping as-is (legacy behavior). Otherwise use training mapping to interpret test labels.')
    parser.add_argument('--save-confusion', action='store_true', help='If set, save confusion matrix CSV and per-class counts to the output directory')
    args = parser.parse_args()

    def find_default_checkpoint():
        base = 'results_stage2_accelerated'
        candidates = [os.path.join(base, 'best_model_finetuned.pth'), os.path.join(base, 'best_model.pth')]
        for c in candidates:
            if os.path.exists(c):
                return c
        # fallback: pick latest .pth in the folder
        files = glob.glob(os.path.join(base, '*.pth'))
        if files:
            files = sorted(files, key=os.path.getmtime, reverse=True)
            return files[0]
        return None

    # Model path (from CLI or auto-select)
    model_path = args.model if args.model else find_default_checkpoint()
    
    # Check if model exists
    if not model_path or not os.path.exists(model_path):
        print("❌ Model file not found: {}".format(model_path))
        print("Please ensure your trained model is saved in the correct location.")
        return
    
    try:
        # Load class names
        class_names = load_class_names('class_names.json')
        # Use 200 classes to match the saved model
        num_classes = 200  # Your model was trained with 200 classes

        # Load model
        model = load_model(model_path, num_classes, device)
        print("Model loaded successfully.")

        # Create loader according to requested split
        if args.split == 'test':
            if getattr(args, 'raw_test_mapping', False) or args.raw_test_mapping:
                # Legacy behaviour: create test loader directly from Test folder
                test_transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                test_dataset = BirdDataset(image_dir=os.path.join('data', 'Test'), annotation_file=os.path.join('data', 'test.txt'), transform=test_transform)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
                class_names = test_dataset.classes
                print("Test data loader created (raw mapping). Number of batches: {}".format(len(test_loader)))
            else:
                # Preferred: interpret test labels using training mapping so indices align with model
                test_loader, class_names = create_test_loader_with_train_mapping(
                    train_dir='data/Train', train_txt='data/train.txt', test_dir='data/Test', test_txt='data/test.txt',
                    batch_size=32, image_size=384, num_workers=0, augmentation_level='advanced'
                )
                print("Test data loader created (using training mapping). Number of batches: {}".format(len(test_loader)))
        else:
            # create_data_loaders returns train_loader, val_loader, test_loader
            print('Creating data loaders to obtain the validation split (matching training transforms)')
            train_dir = 'data/Train'
            train_txt = 'data/train.txt'
            test_dir = 'data/Test'
            test_txt = 'data/test.txt'
            train_loader, val_loader, test_loader2, num_classes_unused, class_names_unused = create_data_loaders(
                train_dir, train_txt, test_dir, test_txt, batch_size=32, image_size=384, num_workers=0, validation_split=0.1, augmentation_level='advanced'
            )
            test_loader = val_loader
            print('Validation loader created. Number of batches:', len(test_loader))

        # Evaluate model
        if args.save_confusion:
            results, preds, targets = evaluate_model(model, test_loader, device, return_preds=True)
        else:
            results = evaluate_model(model, test_loader, device)

        # Print results
        print_results(results, class_names)

        # Save results
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, 'final_evaluation_results.json')
        save_results(results, out_path)

        # Optionally save confusion matrix and per-class CSV
        if args.save_confusion:
            # Ensure preds and targets exist
            try:
                cm = confusion_matrix(targets, preds, labels=sorted(list(set(targets))))
            except Exception:
                cm = confusion_matrix(targets, preds)

            # Save confusion matrix as CSV
            cm_path = os.path.join(args.out_dir, 'confusion_matrix.csv')
            with open(cm_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([''] + [f'class_{i}' for i in range(cm.shape[1])])
                for i, row in enumerate(cm.tolist()):
                    writer.writerow([f'class_{i}'] + row)

            # Save per-class counts and accuracies
            counts_path = os.path.join(args.out_dir, 'per_class_counts.csv')
            with open(counts_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['class_id', 'class_name', 'total', 'correct', 'accuracy'])
                for k, v in results['per_class_details'].items():
                    class_id = int(k)
                    name = class_names[class_id] if class_names and class_id < len(class_names) else f'class_{class_id}'
                    writer.writerow([class_id, name, v['total'], v['correct'], v['accuracy']])

            print(f"Saved confusion matrix to: {cm_path}")
            print(f"Saved per-class counts to: {counts_path}")

        print("\nEVALUATION COMPLETED SUCCESSFULLY!")
        print("Use these metrics in your report:")
        print("   - Top-1 Accuracy: {:.2f}%".format(results['top1_accuracy']))
        print("   - Average Accuracy per Class: {:.2f}%".format(results['average_accuracy_per_class']))

    except Exception as e:
        print("Error during evaluation: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()