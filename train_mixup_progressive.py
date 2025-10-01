"""
Progressive training with MixUp and TTA for 90% accuracy target.
Enhanced version building on 75.50% baseline.
"""
import torch
import torch.nn as nn
import os
import argparse
from train_enhanced_90 import main as train_enhanced
from mixup import MixupCutmix, mixup_loss_fn, TestTimeAugmentation
from data_loader import create_data_loaders, get_data_transforms
from models import create_model
from trainer import ModelTrainer
from metrics import evaluate_model


def train_with_mixup_progressive():
    """Progressive training pipeline with MixUp and TTA."""
    
    # Stage 1: MixUp Training (320px)
    print("🚀 STAGE 1: Enhanced Training with MixUp (320px)")
    print("Target: 80-82% accuracy")
    
    # Enhanced arguments for Stage 1
    stage1_args = {
        'model_type': 'efficientnet_b2',
        'image_size': 320,  # Increased from 288
        'batch_size': 16,   # Reduced for larger images
        'learning_rate': 8e-5,  # Slightly reduced
        'weight_decay': 1e-4,
        'dropout_rate': 0.3,  # Reduced from 0.5
        'augmentation_level': 'heavy',  # Fixed: using valid augmentation level
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './results_stage1_mixup',
        'data_dir': './data'
    }
    
    # Create directories
    os.makedirs(stage1_args['save_dir'], exist_ok=True)
    
    # Load data with stronger augmentation
    print("\nLoading data with enhanced augmentation...")
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        train_dir=os.path.join(stage1_args['data_dir'], 'Train'),
        train_txt=os.path.join(stage1_args['data_dir'], 'train.txt'),
        test_dir=os.path.join(stage1_args['data_dir'], 'Test'),
        test_txt=os.path.join(stage1_args['data_dir'], 'test.txt'),
        batch_size=stage1_args['batch_size'],
        image_size=stage1_args['image_size'],
        num_workers=4,
        validation_split=0.2,
        augmentation_level=stage1_args['augmentation_level']
    )
    
    # Create model
    print(f"\nCreating {stage1_args['model_type']} model...")
    model = create_model(
        num_classes=num_classes,
        model_type=stage1_args['model_type'],
        pretrained=True,
        dropout_rate=stage1_args['dropout_rate']
    )
    
    # Initialize MixUp
    mixup = MixupCutmix(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=0.8,  # High probability for strong augmentation
        switch_prob=0.5
    )
    
    # Custom trainer with MixUp
    trainer = MixUpTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=stage1_args['device'],
        class_names=class_names,
        mixup=mixup
    )
    
    # Stage 1 Training
    print("\n--- Stage 1: MixUp Training ---")
    history = trainer.train(
        num_epochs=50,
        learning_rate=stage1_args['learning_rate'],
        weight_decay=stage1_args['weight_decay'],
        scheduler_type='cosine',
        early_stopping_patience=15,
        save_dir=stage1_args['save_dir']
    )
    
    # Evaluate Stage 1
    print("\n🔍 Evaluating Stage 1...")
    stage1_metrics = evaluate_model(
        model=trainer.model,
        data_loader=test_loader,
        device=stage1_args['device'],
        class_names=class_names,
        criterion=nn.CrossEntropyLoss(),
        save_dir=stage1_args['save_dir']
    )
    
    print(f"Stage 1 Accuracy: {stage1_metrics['top1_accuracy']:.4f} ({stage1_metrics['top1_accuracy']*100:.2f}%)")
    
    # Stage 2: TTA Enhancement
    if stage1_metrics['top1_accuracy'] >= 0.78:  # If Stage 1 successful
        print("\n🎯 STAGE 2: Test Time Augmentation Enhancement")
        print("Target: 85-87% accuracy")
        
        # Initialize TTA
        tta = TestTimeAugmentation(n_tta=8, image_size=stage1_args['image_size'])
        
        # Evaluate with TTA
        print("\n🔍 Evaluating with TTA...")
        tta_metrics = evaluate_with_tta(
            model=trainer.model,
            data_loader=test_loader,
            device=stage1_args['device'],
            class_names=class_names,
            tta=tta,
            save_dir=os.path.join(stage1_args['save_dir'], 'tta_results')
        )
        
        print(f"TTA Accuracy: {tta_metrics['top1_accuracy']:.4f} ({tta_metrics['top1_accuracy']*100:.2f}%)")
        
        # Stage 3: EfficientNet-B3 + Progressive Resizing
        if tta_metrics['top1_accuracy'] >= 0.83:
            print("\n🏆 STAGE 3: EfficientNet-B3 + Progressive Training")
            print("Target: 87-92% accuracy")
            
            stage3_progressive_training(
                base_model_path=os.path.join(stage1_args['save_dir'], 'best_model.pth'),
                class_names=class_names,
                num_classes=num_classes
            )
    
    return stage1_metrics


class MixUpTrainer(ModelTrainer):
    """Enhanced trainer with MixUp support."""
    
    def __init__(self, model, train_loader, val_loader, device, class_names, mixup):
        super().__init__(model, train_loader, val_loader, device, class_names)
        self.mixup = mixup
    
    def train_epoch(self, optimizer, criterion, scheduler=None):
        """Training epoch with MixUp augmentation."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply MixUp/CutMix
            mixed_inputs, mixed_targets = self.mixup(inputs, targets)
            
            optimizer.zero_grad()
            outputs = self.model(mixed_inputs)
            
            # Calculate MixUp loss
            loss = mixup_loss_fn(criterion, outputs, mixed_targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (approximate for mixed targets)
            if isinstance(mixed_targets, dict):
                # For mixed targets, use lambda-weighted accuracy
                _, predicted = outputs.max(1)
                lam = mixed_targets['lam']
                targets_a = mixed_targets['targets_a']
                targets_b = mixed_targets['targets_b']
                
                correct_a = predicted.eq(targets_a).sum().item()
                correct_b = predicted.eq(targets_b).sum().item()
                correct += lam * correct_a + (1 - lam) * correct_b
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(mixed_targets).sum().item()
            
            total += targets.size(0)
            
            if batch_idx % 20 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        if scheduler:
            scheduler.step()
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc


def evaluate_with_tta(model, data_loader, device, class_names, tta, save_dir):
    """Evaluate model with Test Time Augmentation."""
    import torch.nn.functional as F
    from metrics import MetricsCalculator
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    print(f"Running TTA with {tta.n_tta} augmentations...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply TTA
            tta_predictions = tta(model, inputs, device)
            
            # Get final predictions
            _, predicted = tta_predictions.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f'TTA Progress: {batch_idx}/{len(data_loader)}')
    
    # Calculate metrics using MetricsCalculator
    os.makedirs(save_dir, exist_ok=True)
    metrics_calc = MetricsCalculator(class_names)
    
    # Convert to tensors for metrics calculation
    pred_tensor = torch.tensor(all_predictions)
    target_tensor = torch.tensor(all_targets)
    
    metrics_calc.update(pred_tensor, target_tensor)
    metrics = metrics_calc.calculate_metrics()
    
    # Plot confusion matrix
    metrics_calc.plot_confusion_matrix(
        metrics=metrics,
        save_path=os.path.join(save_dir, 'confusion_matrix_tta.png')
    )
    
    return metrics


def stage3_progressive_training(base_model_path, class_names, num_classes):
    """Stage 3: EfficientNet-B3 with progressive resizing."""
    print("\n--- Stage 3: Progressive EfficientNet-B3 Training ---")
    
    # Progressive training stages
    progressive_stages = [
        {'size': 352, 'epochs': 20, 'lr': 5e-5, 'model': 'efficientnet_b3'},
        {'size': 384, 'epochs': 25, 'lr': 3e-5, 'model': 'efficientnet_b3'},
        {'size': 416, 'epochs': 15, 'lr': 1e-5, 'model': 'efficientnet_b3'}
    ]
    
    current_model_path = base_model_path
    
    for stage_idx, stage_config in enumerate(progressive_stages):
        print(f"\nProgressive Stage {stage_idx + 1}: {stage_config['size']}px")
        
        # Load data for current stage
        train_loader, val_loader, test_loader, _, _ = create_data_loaders(
            train_dir='./data/Train',
            train_txt='./data/train.txt',
            test_dir='./data/Test',
            test_txt='./data/test.txt',
            batch_size=12,  # Reduced for larger images
            image_size=stage_config['size'],
            num_workers=4,
            validation_split=0.2,
            augmentation_level='heavy'  # Fixed: using valid augmentation level
        )
        
        # Create/load model
        if stage_idx == 0:
            # Load from Stage 1
            model = create_model(
                num_classes=num_classes,
                model_type=stage_config['model'],
                pretrained=True,
                dropout_rate=0.2
            )
            # Load Stage 1 weights (with size adjustment)
            try:
                checkpoint = torch.load(current_model_path, map_location='cpu')
                model.load_state_dict(checkpoint, strict=False)
                print("Loaded Stage 1 weights")
            except:
                print("Using pretrained weights")
        else:
            # Load from previous progressive stage
            model = torch.load(current_model_path, map_location='cpu')
        
        # Train current stage
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            class_names=class_names
        )
        
        stage_save_dir = f'./results_stage3_progressive_{stage_idx + 1}'
        os.makedirs(stage_save_dir, exist_ok=True)
        
        history = trainer.train(
            num_epochs=stage_config['epochs'],
            learning_rate=stage_config['lr'],
            weight_decay=5e-5,
            scheduler_type='cosine',
            early_stopping_patience=10,
            save_dir=stage_save_dir
        )
        
        # Update model path for next stage
        current_model_path = os.path.join(stage_save_dir, 'best_model.pth')
        
        # Evaluate current stage
        stage_metrics = evaluate_model(
            model=trainer.model,
            data_loader=test_loader,
            device=device,
            class_names=class_names,
            criterion=nn.CrossEntropyLoss(),
            save_dir=stage_save_dir
        )
        
        print(f"Progressive Stage {stage_idx + 1} Accuracy: {stage_metrics['top1_accuracy']:.4f} ({stage_metrics['top1_accuracy']*100:.2f}%)")
        
        # Apply TTA to final stage
        if stage_idx == len(progressive_stages) - 1:
            print("\n🎯 Final TTA Evaluation...")
            tta = TestTimeAugmentation(n_tta=10, image_size=stage_config['size'])
            final_tta_metrics = evaluate_with_tta(
                model=trainer.model,
                data_loader=test_loader,
                device=device,
                class_names=class_names,
                tta=tta,
                save_dir=os.path.join(stage_save_dir, 'final_tta')
            )
            
            print(f"\n🏆 FINAL RESULT: {final_tta_metrics['top1_accuracy']:.4f} ({final_tta_metrics['top1_accuracy']*100:.2f}%)")
            
            if final_tta_metrics['top1_accuracy'] >= 0.90:
                print("🎉 SUCCESS! 90% TARGET ACHIEVED!")
            else:
                print(f"📈 Progress: {final_tta_metrics['top1_accuracy']*100:.2f}% (Target: 90%)")


if __name__ == '__main__':
    train_with_mixup_progressive()