#!/usr/bin/env python3
"""
Enhanced training script for bird classification with advanced techniques.
"""
import os
import argparse
import torch
import torch.nn as nn
from data_loader import create_data_loaders
from models import create_model
from trainer import ModelTrainer, get_loss_function
from metrics import evaluate_model
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Bird Classification Training')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='data/Train',
                       help='Path to training images directory')
    parser.add_argument('--train_txt', type=str, default='data/train.txt',
                       help='Path to training annotations file')
    parser.add_argument('--test_dir', type=str, default='data/Test',
                       help='Path to test images directory')
    parser.add_argument('--test_txt', type=str, default='data/test.txt',
                       help='Path to test annotations file')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='efficientnet_b3',
                       choices=['resnet50', 'resnet18', 'resnet101', 'efficientnet_b0', 
                               'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 
                               'efficientnet_b4', 'lightweight'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=60,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='L2 regularization strength')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Patience for early stopping')
    
    # Advanced training options
    parser.add_argument('--loss_type', type=str, default='focal',
                       choices=['crossentropy', 'focal', 'label_smoothing'],
                       help='Loss function type')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--augmentation_level', type=str, default='advanced',
                       choices=['basic', 'advanced', 'heavy'],
                       help='Data augmentation level')
    
    # Multi-stage training
    parser.add_argument('--multistage_training', action='store_true',
                       help='Use multi-stage training strategy')
    parser.add_argument('--stage1_epochs', type=int, default=15,
                       help='Epochs for stage 1 (frozen backbone)')
    parser.add_argument('--stage2_epochs', type=int, default=30,
                       help='Epochs for stage 2 (full training)')
    parser.add_argument('--stage3_epochs', type=int, default=15,
                       help='Epochs for stage 3 (fine-tuning)')
    
    # Data parameters
    parser.add_argument('--image_size', type=int, default=288,
                       help='Input image size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of training data for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--save_dir', type=str, default='./results_enhanced',
                       help='Directory to save results')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Get the appropriate device."""
    if device_arg == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_arg


def train_single_stage(args, device):
    """Single-stage training approach."""
    print("=== Single-Stage Training ===")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        train_dir=args.train_dir,
        train_txt=args.train_txt,
        test_dir=args.test_dir,
        test_txt=args.test_txt,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        augmentation_level=args.augmentation_level
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
    
    # Create loss function
    if args.loss_type == 'focal':
        criterion = get_loss_function('focal', alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_type == 'label_smoothing':
        criterion = get_loss_function('label_smoothing', smoothing=args.label_smoothing)
    else:
        criterion = get_loss_function('crossentropy')
    
    print(f"Using {args.loss_type} loss function")
    
    # Create trainer
    trainer = ModelTrainer(model, train_loader, val_loader, device, class_names)
    
    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        trainer.load_checkpoint(args.resume_from)
        print(f"Resumed from checkpoint: {args.resume_from}")
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        early_stopping_patience=args.early_stopping_patience,
        criterion=criterion,
        save_dir=os.path.join(args.save_dir, 'checkpoints')
    )
    
    return trainer, test_loader, class_names


def train_multistage(args, device):
    """Multi-stage training approach."""
    print("=== Multi-Stage Training ===")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        train_dir=args.train_dir,
        train_txt=args.train_txt,
        test_dir=args.test_dir,
        test_txt=args.test_txt,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        augmentation_level=args.augmentation_level
    )
    
    print(f"Number of classes: {num_classes}")
    
    # Stage 1: Frozen backbone training
    print("\n--- Stage 1: Training with frozen backbone ---")
    model = create_model(
        num_classes=num_classes,
        model_type=args.model_type,
        pretrained=args.pretrained,
        dropout_rate=args.dropout_rate,
        freeze_backbone=True  # Freeze backbone
    )
    
    criterion = get_loss_function('crossentropy')  # Start with simple loss
    trainer = ModelTrainer(model, train_loader, val_loader, device, class_names)
    
    trainer.train(
        num_epochs=args.stage1_epochs,
        learning_rate=args.learning_rate * 10,  # Higher LR for classifier only
        weight_decay=args.weight_decay,
        scheduler_type='step',
        early_stopping_patience=10,
        criterion=criterion,
        save_dir=os.path.join(args.save_dir, 'stage1_checkpoints')
    )
    
    # Stage 2: Full model training
    print("\n--- Stage 2: Full model training ---")
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Use advanced loss function
    if args.loss_type == 'focal':
        criterion = get_loss_function('focal', alpha=args.focal_alpha, gamma=args.focal_gamma)
    elif args.loss_type == 'label_smoothing':
        criterion = get_loss_function('label_smoothing', smoothing=args.label_smoothing)
    else:
        criterion = get_loss_function('crossentropy')
    
    trainer.train(
        num_epochs=args.stage2_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler_type,
        early_stopping_patience=args.early_stopping_patience,
        criterion=criterion,
        save_dir=os.path.join(args.save_dir, 'stage2_checkpoints')
    )
    
    # Stage 3: Fine-tuning
    print("\n--- Stage 3: Fine-tuning ---")
    trainer.train(
        num_epochs=args.stage3_epochs,
        learning_rate=args.learning_rate * 0.1,  # Lower LR for fine-tuning
        weight_decay=args.weight_decay,
        scheduler_type='plateau',
        early_stopping_patience=args.early_stopping_patience,
        criterion=criterion,
        save_dir=os.path.join(args.save_dir, 'stage3_checkpoints')
    )
    
    return trainer, test_loader, class_names


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if data files exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        return
    
    if not os.path.exists(args.train_txt):
        print(f"Error: Training annotation file not found: {args.train_txt}")
        return
    
    try:
        # Choose training strategy
        if args.multistage_training:
            trainer, test_loader, class_names = train_multistage(args, device)
        else:
            trainer, test_loader, class_names = train_single_stage(args, device)
        
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
            'augmentation_level': args.augmentation_level,
            'loss_type': args.loss_type,
            'image_size': args.image_size,
            'multistage_training': args.multistage_training,
            'test_metrics': {k: v for k, v in test_metrics.items() 
                           if not isinstance(v, (dict, list, tuple))},
            'args': vars(args)
        }
        
        import json
        with open(os.path.join(args.save_dir, 'results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\nEnhanced training complete! Results saved to {args.save_dir}")
        print("\nKey Results:")
        print(f"Test Accuracy: {test_metrics['top1_accuracy']:.4f}")
        print(f"Average Accuracy per Class: {test_metrics['average_accuracy_per_class']:.4f}")
        if 'top3_accuracy' in test_metrics:
            print(f"Top-3 Accuracy: {test_metrics['top3_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()