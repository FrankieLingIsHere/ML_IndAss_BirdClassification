"""
Enhanced training script with progressive training and advanced techniques for 90% accuracy.
"""
import os
import argparse
import torch
import torch.nn as nn
from data_loader import create_data_loaders
from models import create_model
from advanced_trainer import AdvancedModelTrainer, get_progressive_training_config
from metrics import evaluate_model
from mixup import TestTimeAugmentation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Bird Classification Training for 90% Accuracy')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default='data/Train',
                        help='Path to training images directory')
    parser.add_argument('--train_txt', type=str, default='data/train.txt', 
                        help='Path to training annotations file')
    parser.add_argument('--test_dir', type=str, default='data/Test',
                        help='Path to test images directory') 
    parser.add_argument('--test_txt', type=str, default='data/test.txt',
                        help='Path to test annotations file')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='efficientnet_b2',
                        choices=['resnet50', 'resnet18', 'efficientnet_b0', 'efficientnet_b1', 
                                'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for regularization')
    
    # Training arguments
    parser.add_argument('--image_size', type=int, default=288,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--augmentation_level', type=str, default='advanced',
                        choices=['basic', 'advanced', 'heavy'],
                        help='Data augmentation level')
    
    # Progressive training
    parser.add_argument('--progressive_training', action='store_true', default=True,
                        help='Use progressive training strategy')
    parser.add_argument('--custom_stages', type=str, default=None,
                        help='Path to custom training stages JSON file')
    
    # Advanced features
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use Test Time Augmentation')
    parser.add_argument('--tta_iterations', type=int, default=5,
                        help='Number of TTA iterations')
    parser.add_argument('--use_mixup', action='store_true', default=True,
                        help='Use Mixup/CutMix augmentation')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for training')
    parser.add_argument('--save_dir', type=str, default='./enhanced_results',
                        help='Directory to save results')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main enhanced training function."""
    print("🚀 Enhanced Bird Classification Training for 90% Accuracy")
    print("="*70)
    
    args = parse_arguments()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Create data loaders with enhanced augmentation
        print("\nCreating enhanced data loaders...")
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
        
        # Create enhanced model
        print(f"\nCreating enhanced {args.model_type} model...")
        model = create_model(
            num_classes=num_classes,
            model_type=args.model_type,
            pretrained=args.pretrained,
            dropout_rate=args.dropout_rate
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Initialize advanced trainer
        print("\nInitializing advanced trainer with modern techniques...")
        trainer = AdvancedModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_names=class_names
        )
        
        # Load checkpoint if provided
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get training configuration
        if args.progressive_training:
            print("\n🎯 Starting Progressive Training Strategy")
            print("This advanced approach will train the model in multiple stages:")
            print("  Stage 1: Classifier-only training with moderate augmentation")
            print("  Stage 2: End-to-end fine-tuning with heavy augmentation")
            print("  Stage 3: Final optimization with maximum regularization")
            
            if args.custom_stages and os.path.exists(args.custom_stages):
                import json
                with open(args.custom_stages, 'r') as f:
                    training_stages = json.load(f)
            else:
                training_stages = get_progressive_training_config()
            
            # Run progressive training
            history = trainer.progressive_train(
                stages=training_stages,
                save_dir=os.path.join(args.save_dir, 'checkpoints')
            )
            
        else:
            print("\nUsing single-stage training...")
            # Note: Single-stage training not recommended for 90% accuracy target
            from trainer import ModelTrainer
            standard_trainer = ModelTrainer(model, train_loader, val_loader, device, class_names)
            history = standard_trainer.train(
                num_epochs=50,
                learning_rate=1e-4,
                weight_decay=2e-4,
                scheduler_type='cosine',
                early_stopping_patience=15,
                save_dir=os.path.join(args.save_dir, 'checkpoints')
            )
        
        # Plot training history
        print("\nGenerating training visualizations...")
        trainer.plot_training_history(
            save_path=os.path.join(args.save_dir, 'enhanced_training_history.png')
        )
        
        # Enhanced evaluation
        print("\n🔍 Evaluating model performance...")
        
        if args.use_tta:
            print("Running enhanced evaluation with Test Time Augmentation...")
            test_metrics = evaluate_model_with_tta(
                model=trainer.model,
                data_loader=test_loader,
                device=device,
                class_names=class_names,
                tta=TestTimeAugmentation(n_tta=args.tta_iterations, image_size=args.image_size),
                save_dir=args.save_dir
            )
        else:
            # Standard evaluation
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
        
        # Save comprehensive results
        results_summary = {
            'model_type': args.model_type,
            'progressive_training': args.progressive_training,
            'image_size': args.image_size,
            'augmentation_level': args.augmentation_level,
            'use_tta': args.use_tta,
            'use_mixup': args.use_mixup,
            'test_metrics': {k: v for k, v in test_metrics.items() 
                           if not isinstance(v, (dict, list, tuple))},
            'args': vars(args)
        }
        
        import json
        with open(os.path.join(args.save_dir, 'enhanced_results_summary.json'), 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Print final results
        print("\n" + "="*70)
        print("🎉 ENHANCED TRAINING COMPLETE!")
        print("="*70)
        print(f"Results saved to: {args.save_dir}")
        print("\n📊 Final Performance:")
        print(f"  Test Accuracy: {test_metrics['top1_accuracy']:.4f} ({test_metrics['top1_accuracy']*100:.2f}%)")
        print(f"  Average per Class: {test_metrics['average_accuracy_per_class']:.4f} ({test_metrics['average_accuracy_per_class']*100:.2f}%)")
        
        if 'top3_accuracy' in test_metrics:
            print(f"  Top-3 Accuracy: {test_metrics['top3_accuracy']:.4f} ({test_metrics['top3_accuracy']*100:.2f}%)")
        if 'top5_accuracy' in test_metrics:
            print(f"  Top-5 Accuracy: {test_metrics['top5_accuracy']:.4f} ({test_metrics['top5_accuracy']*100:.2f}%)")
        
        print("\n🚀 Performance Analysis:")
        if test_metrics['top1_accuracy'] >= 0.90:
            print("  🎯 EXCELLENT! Target 90% accuracy achieved!")
        elif test_metrics['top1_accuracy'] >= 0.85:
            print("  ✅ GREAT! Very close to 90% target. Consider ensemble methods.")
        elif test_metrics['top1_accuracy'] >= 0.80:
            print("  🔥 GOOD! Solid improvement. Try longer training or ensemble.")
        elif test_metrics['top1_accuracy'] >= 0.75:
            print("  ⭐ PROGRESS! Consider advanced techniques like knowledge distillation.")
        else:
            print("  📈 Keep improving! Try different architectures or more data.")
        
        print("\n💡 Next Steps for Further Improvement:")
        print("  1. Ensemble multiple models (different architectures)")
        print("  2. Knowledge distillation from larger models")
        print("  3. Self-supervised pre-training on bird data")
        print("  4. Advanced architectures (Vision Transformers)")
        print("  5. Progressive resizing with higher resolution")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


def evaluate_model_with_tta(model, data_loader, device, class_names, tta, save_dir):
    """Evaluate model with Test Time Augmentation."""
    from metrics import MetricsCalculator\n    \n    model.eval()\n    metrics_calc = MetricsCalculator(class_names)\n    \n    total_loss = 0\n    num_batches = 0\n    criterion = nn.CrossEntropyLoss()\n    \n    print(\"Running TTA evaluation...\")\n    with torch.no_grad():\n        for inputs, targets in data_loader:\n            inputs, targets = inputs.to(device), targets.to(device)\n            \n            # Get TTA predictions\n            tta_probs = tta(model, inputs, device)\n            predictions = torch.argmax(tta_probs, dim=1)\n            \n            # Calculate loss on original predictions\n            outputs = model(inputs)\n            loss = criterion(outputs, targets)\n            total_loss += loss.item()\n            num_batches += 1\n            \n            # Update metrics with TTA predictions\n            metrics_calc.update(predictions.cpu(), targets.cpu(), tta_probs.cpu())\n    \n    # Calculate final metrics\n    metrics = metrics_calc.calculate_metrics()\n    metrics['loss'] = total_loss / num_batches\n    \n    # Print and plot results\n    metrics_calc.print_metrics(metrics)\n    \n    # Save plots\n    try:\n        metrics_calc.plot_confusion_matrix(metrics, os.path.join(save_dir, 'tta_confusion_matrix.png'))\n        metrics_calc.plot_per_class_metrics(metrics, os.path.join(save_dir, 'tta_per_class_metrics.png'))\n    except Exception as e:\n        print(f\"Warning: Could not save plots: {e}\")\n    \n    return metrics\n\n\nif __name__ == '__main__':\n    main()