"""
Enhanced Stage 2 Training Script - Colab Synchronized Version
Achieving 76.74% accuracy with optimized configuration.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.nn.utils.clip_grad import clip_grad_norm_
import time
from tqdm import tqdm

from data_loader import create_data_loaders
from models import create_model
from trainer import ModelTrainer
from metrics import evaluate_model


def train_stage2_enhanced():
    """
    Stage 2 Enhanced Training - Colab Synchronized
    Target: 76.74% accuracy reproduction
    """
    print("🚀 STAGE 2: Enhanced Training (Colab Synchronized)")
    print("Target: 76.74% accuracy")
    print("="*60)
    
    # Stage 2 Configuration (Colab optimized)
    config = {
        'model_type': 'efficientnet_b2',
        'image_size': 320,      # Stage 2 image size
        'batch_size': 12,       # Optimized batch size
        'learning_rate': 6e-5,  # Stage 2 learning rate
        'weight_decay': 1e-4,
        'dropout_rate': 0.3,    # Stage 2 dropout rate
        'num_epochs': 60,       # Stage 2 epochs
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './results_stage2_accelerated',
        'data_dir': './data',
        'warmup_epochs': 5,     # Stage 2 warmup
        'patience': 12          # Early stopping patience
    }
    
    print(f"Using device: {config['device']}")
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Load data with Stage 2 augmentation
    print("\nLoading data with Stage 2 augmentation...")
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        train_dir=os.path.join(config['data_dir'], 'Train'),
        train_txt=os.path.join(config['data_dir'], 'train.txt'),
        test_dir=os.path.join(config['data_dir'], 'Test'),
        test_txt=os.path.join(config['data_dir'], 'test.txt'),
        batch_size=config['batch_size'],
        image_size=config['image_size'],
        num_workers=4,
        validation_split=0.2,
        augmentation_level='heavy'  # Stage 2 augmentation
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create Stage 2 model
    print(f"\nCreating {config['model_type']} model (Stage 2 configuration)...")
    model = create_model(
        num_classes=num_classes,
        model_type=config['model_type'],
        pretrained=True,
        dropout_rate=config['dropout_rate']
    )
    model.to(config['device'])
    
    # Stage 2 optimizer configuration
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Stage 2 scheduler configuration
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training tracking
    best_val_acc = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\n🎯 Starting Stage 2 Enhanced Training")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Image Size: {config['image_size']}")
    print(f"Dropout Rate: {config['dropout_rate']}")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            train_accuracy = 100 * correct_predictions / total_samples
            train_pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{train_accuracy:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(config['device']), labels.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_acc = 100 * val_correct / val_total
                val_pbar.set_postfix({'Val Acc': f'{val_acc:.2f}%'})
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_accuracy)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
                'config': config
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            
            print(f"✅ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['patience']}")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Test evaluation
    test_accuracy = evaluate_model(model, test_loader, config['device'], class_names)
    print(f"\n🎯 Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final results
    final_results = {
        'test_accuracy': test_accuracy,
        'best_val_accuracy': best_val_acc,
        'training_history': training_history,
        'config': config,
        'total_epochs': epoch + 1
    }
    
    with open(os.path.join(config['save_dir'], 'training_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Copy best model for deployment
    import shutil
    shutil.copy(
        os.path.join(config['save_dir'], 'best_model.pth'),
        './best_model.pth'  # For Hugging Face deployment
    )
    
    print(f"\n✅ Stage 2 Enhanced Training Complete!")
    print(f"📁 Results saved to: {config['save_dir']}")
    print(f"🚀 Model ready for Hugging Face deployment: ./best_model.pth")
    
    return model, test_accuracy, best_val_acc


if __name__ == "__main__":
    model, test_acc, val_acc = train_stage2_enhanced()
    print(f"\n🎯 STAGE 2 RESULTS SUMMARY:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")