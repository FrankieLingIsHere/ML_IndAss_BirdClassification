"""
Simple enhanced training script with key improvements. This script no longer stops at a hard 90% threshold.
"""
import argparse
import os
import torch
import torch.nn as nn

from data_loader import create_data_loaders
from models import create_model
from trainer import ModelTrainer
from metrics import evaluate_model
from training_utils import freeze_batchnorm_stats, unfreeze_batchnorm_stats, gradual_unfreeze


def main():
    """Main enhanced training function."""
    # Parse CLI overrides so users can run in Colab without editing the file
    parser = argparse.ArgumentParser(description='Enhanced training (multi-phase)')
    parser.add_argument('--model-type', default='efficientnet_b4', help='Backbone architecture (efficientnet_b4, efficientnet_b3, resnet50, etc.)')
    parser.add_argument('--image-size', type=int, default=448, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--save-dir', default='./results/train_enhanced_b4', help='Directory to save checkpoints and results')
    parser.add_argument('--epochs', type=int, default=75, help='Total training epochs')
    cli_args = parser.parse_args()

    print("🚀 Enhanced Bird Classification Training")
    print("="*70)

    # Use enhanced defaults for better performance on Colab/GPU
    args = {
        'train_dir': 'data/Train',
        'train_txt': 'data/train.txt',
        'test_dir': 'data/Test',
        'test_txt': 'data/test.txt',
        'model_type': cli_args.model_type,
        'image_size': cli_args.image_size,
        'batch_size': cli_args.batch_size,
        'dropout_rate': 0.4,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'num_epochs': cli_args.epochs,
        'augmentation_level': 'advanced',
        'save_dir': cli_args.save_dir,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Using device: {args['device']}")
    print(f"Model type: {args['model_type']}  Image size: {args['image_size']}  Batch size: {args['batch_size']}")
    
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
        
        # Freeze backbone (safe attribute checks)
        backbone = getattr(model, 'backbone', None)
        features = getattr(model, 'features', None)
        if backbone is not None and hasattr(backbone, 'parameters'):
            for param in backbone.parameters():
                param.requires_grad = False
        elif features is not None and hasattr(features, 'parameters'):
            for param in features.parameters():
                param.requires_grad = False

        # Freeze BatchNorm running stats during head warmup
        bb = backbone or features
        if bb is not None:
            freeze_batchnorm_stats(bb)
        
        # Train classifier only
        history_phase1 = trainer.train(
            num_epochs=15,
            learning_rate=5e-4,  # Moderate LR for classifier warmup
            weight_decay=args['weight_decay'],
            scheduler_type='cosine',
            early_stopping_patience=3,
            save_dir=os.path.join(args['save_dir'], 'phase1_checkpoints')
        )
        
        # Phase 2: End-to-end fine-tuning
        print("\n--- Phase 2: End-to-End Fine-tuning ---")
        
        # Gradual unfreeze: unfreeze last blocks first, then full unfreeze
        print('Gradually unfreezing backbone blocks...')
        unfrozen = gradual_unfreeze(model, backbone_attr='backbone', block_name_pattern=r'_blocks\\.\\d+', unfreeze_last_n_blocks=2)
        if unfrozen:
            print('Unfrozen params:', len(unfrozen))
        else:
            # fallback to unfreeze all if block detection fails
            for param in model.parameters():
                param.requires_grad = True
            # Re-enable BatchNorm stats
            bb = getattr(model, 'backbone', None) or getattr(model, 'features', None)
            if bb is not None:
                unfreeze_batchnorm_stats(bb)
        
        # Fine-tune entire model
        history_phase2 = trainer.train(
            num_epochs=40,
            learning_rate=args['learning_rate'],  # Lower LR for fine-tuning
            weight_decay=args['weight_decay'],
            scheduler_type='cosine',
            early_stopping_patience=3,
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
        # Neutral performance categories (no hard target enforced)
        if test_metrics['top1_accuracy'] >= 0.85:
            print("  ✅ STRONG PERFORMANCE: further tuning/ensemble may yield additional gains.")
        elif test_metrics['top1_accuracy'] >= 0.80:
            print("  🔥 GOOD: solid results; consider ensembling or stronger augmentations.")
        elif test_metrics['top1_accuracy'] >= 0.75:
            print("  ⭐ PROGRESS: reasonable baseline; try focal/oversampling for underperforming classes.")
        else:
            print("  📈 Room for improvement. Consider data cleaning, augmentation, or a larger backbone.")

        improvement = test_metrics['top1_accuracy'] - 0.5457  # From baseline
        print(f"  📈 Improvement: +{improvement:.4f} ({improvement*100:.2f}%) from baseline")

        print("\n💡 Key Enhancements Applied:")
        print("  ✅ EfficientNet-B4 architecture")
        print("  ✅ Enhanced dropout (0.5)")
        print("  ✅ Advanced data augmentation")
        print("  ✅ Multi-phase training strategy")
        print("  ✅ Optimized hyperparameters")
        print("  ✅ Fixed validation transform bug")

        # Helpful next-step techniques to improve performance (no hard target enforced)
        print("\n🔧 Useful next steps to try:")
        print("  • Test Time Augmentation (TTA)")
        print("  • Model ensembling and weighted averaging")
        print("  • Knowledge distillation")
        print("  • Progressive resizing")
        print("  • Mixup/CutMix augmentation")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()