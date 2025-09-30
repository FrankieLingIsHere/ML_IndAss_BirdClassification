"""
Advanced trainer with progressive training and modern techniques for 90% accuracy.
"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from mixup import MixupCutmix, mixup_loss_fn, TestTimeAugmentation
from trainer import EarlyStopping


class AdvancedModelTrainer:
    """
    Advanced model trainer with progressive training and modern techniques.
    """
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 device: str = 'cuda', class_names: Optional[List[str]] = None):
        """
        Initialize advanced trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            class_names: List of class names
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.class_names = class_names or [f"class_{i}" for i in range(200)]
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'stage_boundaries': []  # Track progressive training stages
        }
        
        # Advanced augmentation
        self.mixup_cutmix = MixupCutmix(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)
        self.tta = TestTimeAugmentation(n_tta=5)
        
    def progressive_train(self, stages: List[Dict], save_dir: str = "./checkpoints") -> Dict:
        """
        Progressive training with multiple stages and increasing complexity.
        
        Args:
            stages: List of training stage configurations
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        print(f"Starting progressive training with {len(stages)} stages...")
        
        best_val_acc = 0.0
        total_epochs = 0
        
        for stage_idx, stage_config in enumerate(stages):
            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage_config.get('name', f'Stage {stage_idx + 1}')}")
            print(f"{'='*60}")
            
            # Mark stage boundary
            self.history['stage_boundaries'].append(len(self.history['train_loss']))
            
            # Configure stage-specific parameters
            stage_epochs = stage_config.get('epochs', 20)
            stage_lr = stage_config.get('learning_rate', 1e-4)
            stage_wd = stage_config.get('weight_decay', 1e-4)
            scheduler_type = stage_config.get('scheduler_type', 'cosine')
            early_stopping_patience = stage_config.get('early_stopping_patience', 10)
            mixup_prob = stage_config.get('mixup_prob', 0.5)
            freeze_backbone = stage_config.get('freeze_backbone', False)
            
            # Update mixup probability
            self.mixup_cutmix.prob = mixup_prob
            
            # Freeze/unfreeze backbone
            self._configure_backbone_training(freeze_backbone)
            
            # Setup optimizer for this stage
            if freeze_backbone:
                # Only train classifier
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            else:
                trainable_params = self.model.parameters()
                
            optimizer = optim.AdamW(trainable_params, lr=stage_lr, weight_decay=stage_wd)
            
            # Setup scheduler
            scheduler = self._get_scheduler(optimizer, scheduler_type, stage_epochs)
            
            # Setup loss and early stopping
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
            
            # Train this stage
            stage_best_acc = self._train_stage(
                stage_epochs, optimizer, criterion, scheduler, early_stopping,
                stage_idx, save_dir
            )
            
            best_val_acc = max(best_val_acc, stage_best_acc)
            total_epochs += stage_epochs
            
            print(f"Stage {stage_idx + 1} complete. Best accuracy: {stage_best_acc:.4f}")
        
        print(f"\nProgressive training complete! Total epochs: {total_epochs}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history
    
    def _train_stage(self, num_epochs: int, optimizer, criterion, scheduler, 
                    early_stopping, stage_idx: int, save_dir: str) -> float:
        """Train a single progressive stage."""
        stage_best_acc = 0.0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase with advanced augmentation
            train_loss, train_acc = self._train_epoch_advanced(criterion, optimizer)
            
            # Validation phase with TTA
            val_loss, val_acc = self._validate_epoch_tta(criterion)
            
            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f'Stage {stage_idx+1} Epoch {epoch+1:3d}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                  f'LR: {current_lr:.2e} | Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > stage_best_acc:
                stage_best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'stage': stage_idx,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, f'best_model_stage_{stage_idx}.pth'))
            
            # Early stopping check
            if early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1} in stage {stage_idx+1}")
                break
        
        return stage_best_acc
    
    def _train_epoch_advanced(self, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch with advanced augmentation."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply mixup/cutmix
            mixed_inputs, mixed_targets = self.mixup_cutmix(inputs, targets)
            mixed_inputs = mixed_inputs.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(mixed_inputs)
            
            # Calculate loss (handle mixed targets)
            loss = mixup_loss_fn(criterion, outputs, mixed_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics (use original targets for accuracy)
            if isinstance(mixed_targets, dict):
                # For mixup/cutmix, use the dominant class for accuracy calculation
                dominant_targets = mixed_targets['targets_a']
            else:
                dominant_targets = mixed_targets
                
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += dominant_targets.size(0)
            correct += (predicted == dominant_targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch_tta(self, criterion) -> Tuple[float, float]:
        """Validate for one epoch with Test Time Augmentation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Apply TTA
                tta_predictions = self.tta(self.model, inputs, self.device)
                
                # Calculate loss on original predictions
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                
                # Use TTA predictions for accuracy
                _, predicted = torch.max(tta_predictions, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _configure_backbone_training(self, freeze_backbone: bool):
        """Configure backbone training (freeze/unfreeze)."""
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'parameters'):
            for param in self.model.backbone.parameters():
                param.requires_grad = not freeze_backbone
        elif hasattr(self.model, 'features') and hasattr(self.model.features, 'parameters'):
            for param in self.model.features.parameters():
                param.requires_grad = not freeze_backbone
        
        if freeze_backbone:
            print("Backbone frozen - training classifier only")
        else:
            print("Backbone unfrozen - training end-to-end")
    
    def _get_scheduler(self, optimizer, scheduler_type: str, num_epochs: int):
        """Get learning rate scheduler."""
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        elif scheduler_type == 'step':
            return StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        else:
            return optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)  # No scheduling
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot enhanced training history with stage boundaries."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plots
        ax1.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss', alpha=0.7, markersize=3)
        ax1.plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss', alpha=0.7, markersize=3)
        
        # Add stage boundaries
        for boundary in self.history['stage_boundaries'][1:]:  # Skip first boundary at 0
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        ax1.set_title('Model Loss (Progressive Training)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plots
        ax2.plot(epochs, self.history['train_acc'], 'bo-', label='Training Accuracy', alpha=0.7, markersize=3)
        ax2.plot(epochs, self.history['val_acc'], 'ro-', label='Validation Accuracy', alpha=0.7, markersize=3)
        
        # Add stage boundaries
        for boundary in self.history['stage_boundaries'][1:]:
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        ax2.set_title('Model Accuracy (Progressive Training)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], 'go-', alpha=0.7, markersize=3)
        for boundary in self.history['stage_boundaries'][1:]:
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Overfitting indicator
        if len(self.history['train_loss']) > 1:
            overfitting_gap = [val - train for train, val in zip(self.history['train_acc'], self.history['val_acc'])]
            ax4.plot(epochs, overfitting_gap, 'mo-', alpha=0.7, markersize=3)
            for boundary in self.history['stage_boundaries'][1:]:
                ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title('Overfitting Indicator (Val Acc - Train Acc)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy Difference')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()


def get_progressive_training_config():
    """
    Get recommended progressive training configuration for 90% accuracy.
    
    Returns:
        List of training stage configurations
    """
    return [
        {
            'name': 'Stage 1: Classifier Training',
            'epochs': 25,
            'learning_rate': 1e-3,
            'weight_decay': 2e-4,
            'scheduler_type': 'cosine',
            'early_stopping_patience': 10,
            'mixup_prob': 0.3,
            'freeze_backbone': True  # Only train classifier
        },
        {
            'name': 'Stage 2: End-to-End Fine-tuning',
            'epochs': 30,
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'scheduler_type': 'cosine',
            'early_stopping_patience': 15,
            'mixup_prob': 0.5,
            'freeze_backbone': False  # Train full model
        },
        {
            'name': 'Stage 3: Final Optimization',
            'epochs': 20,
            'learning_rate': 1e-5,
            'weight_decay': 5e-5,
            'scheduler_type': 'plateau',
            'early_stopping_patience': 10,
            'mixup_prob': 0.7,
            'freeze_backbone': False
        }
    ]