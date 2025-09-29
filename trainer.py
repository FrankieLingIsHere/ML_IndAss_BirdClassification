"""
Training utilities with overfitting prevention techniques.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in metric to qualify as an improvement
            restore_best_weights: Whether to restore best model weights
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model being trained
            
        Returns:
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print(f"Early stopping triggered. Restored best weights.")
            return True
        
        return False


class ModelTrainer:
    """Comprehensive model trainer with overfitting prevention."""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 device: str = 'cuda', class_names: List[str] = None):
        """
        Initialize trainer.
        
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
        self.class_names = class_names or []
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def train(self, num_epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, scheduler_type: str = 'cosine',
              early_stopping_patience: int = 15, class_weights: Optional[torch.Tensor] = None,
              gradient_clip_value: float = None, save_dir: str = "./checkpoints") -> Dict:
        """
        Train the model with comprehensive overfitting prevention.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            scheduler_type: Learning rate scheduler ('cosine', 'step', 'plateau', 'none')
            early_stopping_patience: Patience for early stopping
            class_weights: Weights for handling class imbalance
            gradient_clip_value: Value for gradient clipping
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer with L2 regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Setup learning rate scheduler
        scheduler = self._get_scheduler(optimizer, scheduler_type, num_epochs)
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
        
        # Training loop
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(criterion, optimizer, gradient_clip_value)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(criterion)
            
            # Update learning rate
            if scheduler_type == 'plateau':
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
            print(f'Epoch {epoch+1:3d}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
                  f'LR: {current_lr:.2e} | Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': self.history,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # Early stopping check
            if early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_acc': best_val_acc,
            'class_names': self.class_names,
        }, os.path.join(save_dir, 'final_model.pth'))
        
        return self.history
    
    def _train_epoch(self, criterion, optimizer, gradient_clip_value: float = None) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_value)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch(self, criterion) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def _get_scheduler(self, optimizer, scheduler_type: str, num_epochs: int):
        """Get learning rate scheduler."""
        if scheduler_type == 'cosine':
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'plateau':
            return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        elif scheduler_type == 'none':
            return lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'bo-', label='Training Loss', alpha=0.7)
        ax1.plot(epochs, self.history['val_loss'], 'ro-', label='Validation Loss', alpha=0.7)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plots
        ax2.plot(epochs, self.history['train_acc'], 'bo-', label='Training Accuracy', alpha=0.7)
        ax2.plot(epochs, self.history['val_acc'], 'ro-', label='Validation Accuracy', alpha=0.7)
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, self.history['learning_rates'], 'go-', alpha=0.7)
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Loss difference (overfitting indicator)
        loss_diff = [val - train for train, val in zip(self.history['train_loss'], self.history['val_loss'])]
        ax4.plot(epochs, loss_diff, 'mo-', alpha=0.7)
        ax4.set_title('Validation - Training Loss (Overfitting Indicator)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Loaded checkpoint from {checkpoint_path}")


def calculate_class_weights(train_loader, num_classes: int, device: str = 'cuda') -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        device: Device to put weights on
        
    Returns:
        Class weights tensor
    """
    class_counts = torch.zeros(num_classes)
    
    for _, targets in train_loader:
        for target in targets:
            class_counts[target] += 1
    
    # Inverse frequency weighting
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    
    # Handle zero counts
    class_weights[class_counts == 0] = 0.0
    
    return class_weights.to(device)