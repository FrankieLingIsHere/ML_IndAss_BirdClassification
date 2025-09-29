#!/usr/bin/env python3
"""
Main training script for bird classification model.
"""
import os
import argparse
import torch
import torch.nn as nn
from data_loader import create_data_loaders
from models import create_model
from trainer import ModelTrainer, calculate_class_weights
from metrics import evaluate_model
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Bird Classification Model')
    
    # Data paths
    parser.add_argument('--train_zip', type=str, default='Train.zip',
                       help='Path to training images zip file')
    parser.add_argument('--train_txt', type=str, default='train.txt',
                       help='Path to training annotations file')
    parser.add_argument('--test_zip', type=str, default='Test.zip',
                       help='Path to test images zip file')
    parser.add_argument('--test_txt', type=str, default='test.txt',
                       help='Path to test annotations file')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet50', 'resnet18', 'efficientnet_b0', 'lightweight'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='L2 regularization strength')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Patience for early stopping')
    parser.add_argument('--gradient_clip_value', type=float, default=None,
                       help='Gradient clipping value')
    
    # Data parameters
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of training data for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights to handle imbalance')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate pre-trained model')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint to load')
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Get the appropriate device."""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if data files exist
    if not os.path.exists(args.train_zip):
        print(f"Warning: Training zip file not found: {args.train_zip}")
        print("Please make sure the Train.zip file is in the current directory.")
        print("Creating dummy data structure for demonstration...")
        create_dummy_data()
        return
    
    if not os.path.exists(args.train_txt):
        print(f"Warning: Training annotation file not found: {args.train_txt}")
        return
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
            train_zip=args.train_zip,
            train_txt=args.train_txt,
            test_zip=args.test_zip,
            test_txt=args.test_txt,
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_workers=args.num_workers,
            validation_split=args.validation_split
        )
        
        print(f"Number of classes: {num_classes}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Create model
        print(f"Creating {args.model_type} model...")
        model = create_model(
            num_classes=num_classes,
            model_type=args.model_type,
            pretrained=args.pretrained,
            dropout_rate=args.dropout_rate,
            freeze_backbone=args.freeze_backbone
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Calculate class weights if requested
        class_weights = None
        if args.use_class_weights:
            print("Calculating class weights...")
            class_weights = calculate_class_weights(train_loader, num_classes, device)
            print("Class weights:", class_weights.cpu().numpy())
        
        # Create trainer
        trainer = ModelTrainer(model, train_loader, val_loader, device, class_names)
        
        # Load checkpoint if provided
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            trainer.load_checkpoint(args.checkpoint_path)
        
        if not args.evaluate_only:
            # Train model
            print("Starting training...")
            history = trainer.train(
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                scheduler_type=args.scheduler_type,
                early_stopping_patience=args.early_stopping_patience,
                class_weights=class_weights,
                gradient_clip_value=args.gradient_clip_value,
                save_dir=os.path.join(args.save_dir, 'checkpoints')
            )
            
            # Plot training history
            print("Plotting training history...")
            trainer.plot_training_history(os.path.join(args.save_dir, 'training_history.png'))
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = evaluate_model(
            model=trainer.model,
            data_loader=test_loader,
            device=device,
            class_names=class_names,
            criterion=nn.CrossEntropyLoss(),
            plot_results=True,
            save_plots=True,
            save_dir=args.save_dir
        )
        
        # Save final results
        results_summary = {
            'model_type': args.model_type,
            'num_classes': num_classes,
            'class_names': class_names,
            'test_metrics': {k: v for k, v in test_metrics.items() 
                           if not isinstance(v, (dict, list, tuple))},
            'args': vars(args)
        }
        
        import json
        with open(os.path.join(args.save_dir, 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\nTraining complete! Results saved to {args.save_dir}")
        print("\nKey Results:")
        print(f"Top-1 Accuracy: {test_metrics['top1_accuracy']:.4f}")
        print(f"Average Accuracy per Class: {test_metrics['average_accuracy_per_class']:.4f}")
        if 'top3_accuracy' in test_metrics:
            print(f"Top-3 Accuracy: {test_metrics['top3_accuracy']:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all data files are present.")
        create_dummy_data()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


def create_dummy_data():
    """Create dummy data structure for demonstration."""
    print("\nCreating dummy data structure for demonstration...")
    
    # Create dummy annotation files
    dummy_classes = ['Cardinal', 'BlueJay', 'Robin', 'Sparrow', 'Eagle']
    
    # Create train.txt
    with open('train.txt', 'w') as f:
        for i in range(100):
            class_name = dummy_classes[i % len(dummy_classes)]
            f.write(f"train_image_{i:03d}.jpg {class_name}\n")
    
    # Create test.txt
    with open('test.txt', 'w') as f:
        for i in range(50):
            class_name = dummy_classes[i % len(dummy_classes)]
            f.write(f"test_image_{i:03d}.jpg {class_name}\n")
    
    print("Created dummy annotation files:")
    print("- train.txt (100 samples)")
    print("- test.txt (50 samples)")
    print("\nTo use this system with real data:")
    print("1. Place your Train.zip file in the current directory")
    print("2. Place your Test.zip file in the current directory")
    print("3. Ensure train.txt and test.txt follow the format: 'image_name class_label'")
    print("4. Run the training script again")


if __name__ == '__main__':
    main()