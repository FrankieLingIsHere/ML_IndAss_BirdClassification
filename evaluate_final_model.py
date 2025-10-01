"""
Final Model Evaluation Script for Bird Classification
Calculates Top-1 accuracy and Average accuracy per class as required.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from collections import defaultdict
from models import BirdClassifier
from data_loader import BirdDataset
import torchvision.transforms as transforms

def load_model(model_path, num_classes=200, device='cpu'):
    """Load the trained model"""
    print("Loading model from: {}".format(model_path))
    
    # Create model with EfficientNet-B3 architecture (as used in training)
    model = BirdClassifier(
        num_classes=num_classes, 
        architecture='efficientnet_b3',
        pretrained=False,  # We're loading trained weights
        dropout_rate=0.3   # Stage 2 dropout rate
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def create_test_loader(data_dir='data', batch_size=32):
    """Create test data loader"""
    # Use the same transforms as validation (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),  # EfficientNet-B3 optimal size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dir = os.path.join(data_dir, 'Test')
    test_annotation = os.path.join(data_dir, 'test.txt')
    
    test_dataset = BirdDataset(
        image_dir=test_dir,
        annotation_file=test_annotation,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    return test_loader

def evaluate_model(model, test_loader, device):
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
    print("\n📊 PRIMARY EVALUATION METRICS:")
    print("="*40)
    print("Top-1 Accuracy: {:.2f}%".format(results['top1_accuracy']))
    print("Average Accuracy per Class: {:.2f}%".format(results['average_accuracy_per_class']))
    
    # Additional details
    print("\n📈 DETAILED STATISTICS:")
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
        
        print("\n🏆 TOP 5 BEST PERFORMING CLASSES:")
        print("="*40)
        for i, (class_id, details) in enumerate(sorted_classes[:5]):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else "Class_{}".format(class_id)
            print("{}. {}: {:.2f}% ({}/{})".format(i+1, class_name, details['accuracy'], details['correct'], details['total']))
        
        print("\n⚠️  TOP 5 WORST PERFORMING CLASSES:")
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
    
    print("\n💾 Results saved to: {}".format(filepath))

def main():
    """Main evaluation function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    
    # Model path (adjust if needed)
    model_path = 'results_stage2_accelerated/best_model.pth'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model file not found: {}".format(model_path))
        print("Please ensure your trained model is saved in the correct location.")
        return
    
    try:
        # Load class names
        class_names = load_class_names('class_names.json')
        # Use 200 classes to match the saved model
        num_classes = 200  # Your model was trained with 200 classes
        
        # Load model
        model = load_model(model_path, num_classes, str(device))
        print("✅ Model loaded successfully!")
        
        # Create test data loader
        test_loader = create_test_loader(data_dir='data', batch_size=32)
        print("✅ Test data loader created. Number of batches: {}".format(len(test_loader)))
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device)
        
        # Print results
        print_results(results, class_names)
        
        # Save results
        save_results(results, 'final_evaluation_results.json')
        
        print("\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("📋 Use these metrics in your report:")
        print("   - Top-1 Accuracy: {:.2f}%".format(results['top1_accuracy']))
        print("   - Average Accuracy per Class: {:.2f}%".format(results['average_accuracy_per_class']))
        
    except Exception as e:
        print("❌ Error during evaluation: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()