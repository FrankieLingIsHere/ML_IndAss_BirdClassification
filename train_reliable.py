"""
Reliable training script that starts with proven ResNet50 architecture.
"""
import os
import torch
import torch.nn as nn

from data_loader import create_data_loaders
from models import create_model
from trainer import ModelTrainer
from metrics import evaluate_model


def main():
    """Reliable training function with proven architecture."""
    print("🛡️ Reliable Bird Classification Training (ResNet50 Base)")
    print("="*70)
    
    # Conservative, proven configuration
    args = {
        'train_dir': 'data/Train',
        'train_txt': 'data/train.txt',
        'test_dir': 'data/Test',
        'test_txt': 'data/test.txt',
        'model_type': 'resnet50',  # Proven to work (54.57% baseline)
        'image_size': 224,  # Standard size
        'batch_size': 16,   
        'dropout_rate': 0.3,  # Conservative dropout
        'learning_rate': 1e-4,  # Standard learning rate
        'weight_decay': 1e-4,   
        'num_epochs': 50,       
        'augmentation_level': 'basic',  # Start simple
        'save_dir': './reliable_results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {args['device']}")
    os.makedirs(args['save_dir'], exist_ok=True)
    
    try:
        # Create data loaders
        print("\nCreating data loaders...")
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
        
        # Create model
        print(f"\nCreating {args['model_type']} model...")
        model = create_model(
            num_classes=num_classes,
            model_type=args['model_type'],
            pretrained=True,
            dropout_rate=args['dropout_rate']
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {param_count:,} (trainable: {trainable_count:,})")
        
        # Test model forward pass
        print("Testing model forward pass...")
        test_input = torch.randn(2, 3, args['image_size'], args['image_size']).to(args['device'])
        model = model.to(args['device'])
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✅ Forward pass successful: {test_output.shape}")
        print(f"Output range: {test_output.min().item():.3f} to {test_output.max().item():.3f}")
        
        # Initialize trainer
        print("\nInitializing trainer...")
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args['device'],
            class_names=class_names
        )
        
        # Single-phase reliable training
        print(f"\n🚀 Starting reliable training for {args['num_epochs']} epochs")
        print("Expected: 15-25% accuracy after 10 epochs")
        
        history = trainer.train(
            num_epochs=args['num_epochs'],
            learning_rate=args['learning_rate'],
            weight_decay=args['weight_decay'],
            scheduler_type='step',  # Simple scheduler
            early_stopping_patience=15,
            save_dir=os.path.join(args['save_dir'], 'checkpoints')
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
            'reliable_training': True,
            'test_accuracy': test_metrics['top1_accuracy'],
            'average_per_class': test_metrics['average_accuracy_per_class'],
            'args': args
        }
        
        import json
        with open(os.path.join(args['save_dir'], 'results_summary.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print results
        print("\n" + "="*70)
        print("🛡️ RELIABLE TRAINING COMPLETE!")
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
        
        # Compare with baseline
        baseline = 54.57  # Your previous best
        if acc_pct > 30:
            print(f"\n🎉 SUCCESS! Model is learning properly!")
            if acc_pct > baseline:
                print(f"  📈 Improvement over baseline: +{acc_pct - baseline:.2f}%")
            print(f"  🎯 Ready for enhanced training techniques")
        elif acc_pct > 15:
            print(f"\n⚡ PROGRESS! Model is learning, but needs tuning")
            print(f"  📊 Reasonable learning progress achieved")
        else:
            print(f"\n⚠️ ISSUE: Still low accuracy, check data or model")
        
        print("\n💡 Next Steps:")
        if acc_pct > 30:
            print("  ✅ Switch to EfficientNet-B2 with enhanced training")
            print("  ✅ Use train_enhanced_90.py with fixed parameters")
            print("  ✅ Apply advanced augmentation and techniques")
        else:
            print("  🔍 Debug data loading and preprocessing")
            print("  🔍 Check class distribution and annotation files")
            print("  🔍 Verify image loading is working correctly")
        
        return acc_pct
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == '__main__':
    main()