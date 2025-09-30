"""
Simple enhanced training script for 90% accuracy with key improvements.
"""
import argparse
import os
import torch
import torch.nn as nn

from data_loader import create_data_loaders
from models import create_model
from trainer import ModelTrainer
from metrics import evaluate_model


def main():
    """Main enhanced training function."""
    print("🚀 Enhanced Bird Classification Training for 90% Accuracy")
    print("="*70)
    
    # Use enhanced defaults for better performance
    args = {
        'train_dir': 'data/Train',
        'train_txt': 'data/train.txt',
        'test_dir': 'data/Test',
        'test_txt': 'data/test.txt',
        'model_type': 'efficientnet_b2',  # Better balance than B3
        'image_size': 288,  # Larger images for more detail
        'batch_size': 16,   # Reduced for stability
        'dropout_rate': 0.4,  # Reduced from 0.5 for better learning
        'learning_rate': 1e-4,  # Increased from 5e-5
        'weight_decay': 1e-4,   # Reduced for initial learning
        'num_epochs': 75,       # More epochs
        'augmentation_level': 'advanced',
        'save_dir': './enhanced_results_90',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {args['device']}")
    os.makedirs(args['save_dir'], exist_ok=True)
    
    try:
        # Create enhanced data loaders
        print("\nCreating enhanced data loaders...")
        train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
            train_dir=args['train_dir'],
            train_txt=args['train_txt'],
            test_dir=args['test_dir'],
            test_txt=args['test_txt'],
            batch_size=args['batch_size'],
            image_size=args['image_size'],
            num_workers=4,
            validation_split=0.2,
            augmentation_level=args['augmentation_level']
        )
        
        print(f"Number of classes: {num_classes}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Create enhanced model
        print(f"\nCreating {args['model_type']} model with enhanced regularization...")
        model = create_model(
            num_classes=num_classes,
            model_type=args['model_type'],
            pretrained=True,
            dropout_rate=args['dropout_rate']
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {param_count:,} (trainable: {trainable_count:,})")
        
        # Initialize trainer
        print("\nInitializing enhanced trainer...")
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args['device'],
            class_names=class_names
        )
        
        # Enhanced training with multiple phases
        print("\n🎯 Starting Enhanced Multi-Phase Training")
        print("Phase 1: Classifier warm-up (15 epochs)")
        print("Phase 2: End-to-end training (40 epochs)")
        
        # Phase 1: Warm up classifier with frozen backbone
        print("\n--- Phase 1: Classifier Warm-up ---")
        
        # Freeze backbone
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif hasattr(model, 'features'):
            for param in model.features.parameters():
                param.requires_grad = False
        
        # Train classifier only
        history_phase1 = trainer.train(
            num_epochs=15,
            learning_rate=5e-4,  # Moderate LR for classifier warmup
            weight_decay=args['weight_decay'],
            scheduler_type='cosine',
            early_stopping_patience=8,
            save_dir=os.path.join(args['save_dir'], 'phase1_checkpoints')
        )
        
        # Phase 2: End-to-end fine-tuning
        print("\n--- Phase 2: End-to-End Fine-tuning ---")
        
        # Unfreeze backbone
        for param in model.parameters():
            param.requires_grad = True
        
        # Fine-tune entire model
        history_phase2 = trainer.train(
            num_epochs=40,
            learning_rate=args['learning_rate'],  # Lower LR for fine-tuning
            weight_decay=args['weight_decay'],
            scheduler_type='cosine',
            early_stopping_patience=12,
            save_dir=os.path.join(args['save_dir'], 'phase2_checkpoints')
        )
        
        # Plot training history
        print("\nGenerating training visualizations...")
        trainer.plot_training_history(
            save_path=os.path.join(args['save_dir'], 'training_history.png')
        )
        
        # Evaluate model
        print("\n🔍 Evaluating final model...")
        test_metrics = evaluate_model(
            model=trainer.model,
            data_loader=test_loader,
            device=args['device'],
            class_names=class_names,
            criterion=nn.CrossEntropyLoss(),
            plot_results=True,
            save_plots=True,
            save_dir=args['save_dir']
        )
        
        # Save results
        results = {
            'model_type': args['model_type'],
            'image_size': args['image_size'],
            'augmentation_level': args['augmentation_level'],
            'multi_phase_training': True,
            'test_accuracy': test_metrics['top1_accuracy'],
            'average_per_class': test_metrics['average_accuracy_per_class'],
            'args': args
        }
        
        import json
        with open(os.path.join(args['save_dir'], 'results_summary.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print results
        print("\n" + "="*70)
        print("🎉 ENHANCED TRAINING COMPLETE!")
        print("="*70)
        print(f"Results saved to: {args['save_dir']}")
        print("\n📊 Final Performance:")
        acc_pct = test_metrics['top1_accuracy'] * 100
        avg_pct = test_metrics['average_accuracy_per_class'] * 100
        print(f"  Test Accuracy: {test_metrics['top1_accuracy']:.4f} ({acc_pct:.2f}%)")
        print(f"  Average per Class: {test_metrics['average_accuracy_per_class']:.4f} ({avg_pct:.2f}%)")
        
        if 'top3_accuracy' in test_metrics:
            top3_pct = test_metrics['top3_accuracy'] * 100
            print(f"  Top-3 Accuracy: {test_metrics['top3_accuracy']:.4f} ({top3_pct:.2f}%)")
        
        print("\n🚀 Performance Analysis:")
        if test_metrics['top1_accuracy'] >= 0.90:
            print("  🎯 EXCELLENT! Target 90% accuracy achieved!")
        elif test_metrics['top1_accuracy'] >= 0.85:
            print("  ✅ GREAT! Very close to 90% target.")
        elif test_metrics['top1_accuracy'] >= 0.80:
            print("  🔥 GOOD! Solid improvement achieved.")
        elif test_metrics['top1_accuracy'] >= 0.75:
            print("  ⭐ PROGRESS! Moving in the right direction.")
        else:
            print("  📈 Room for improvement. Consider ensemble methods.")
        
        improvement = test_metrics['top1_accuracy'] - 0.5457  # From baseline
        print(f"  📈 Improvement: +{improvement:.4f} ({improvement*100:.2f}%) from baseline")
        
        print("\n💡 Key Enhancements Applied:")
        print("  ✅ EfficientNet-B2 architecture")
        print("  ✅ Enhanced dropout (0.5)")
        print("  ✅ Advanced data augmentation")
        print("  ✅ Multi-phase training strategy")
        print("  ✅ Optimized hyperparameters")
        print("  ✅ Fixed validation transform bug")
        
        if test_metrics['top1_accuracy'] < 0.90:
            print("\n🔧 Additional Techniques for 90%+:")
            print("  • Test Time Augmentation (TTA)")
            print("  • Model ensembling")
            print("  • Knowledge distillation")
            print("  • Progressive resizing")
            print("  • Mixup/CutMix augmentation")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()