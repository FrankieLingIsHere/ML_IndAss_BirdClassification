"""
Evaluation metrics for bird classification model.
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import classification_report, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple


class MetricsCalculator:
    """Calculate and track various evaluation metrics."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, probabilities: torch.Tensor = None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Model predictions (class indices)
            targets: Ground truth labels
            probabilities: Class probabilities (for top-k accuracy)
        """
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        if not self.all_predictions:
            return {}
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        metrics = {}
        
        # Top-1 Accuracy (Overall accuracy)
        metrics['top1_accuracy'] = accuracy_score(targets, predictions)
        
        # Per-class accuracy and average accuracy per class
        per_class_accuracy = self._calculate_per_class_accuracy(predictions, targets)
        metrics['per_class_accuracy'] = per_class_accuracy
        metrics['average_accuracy_per_class'] = np.mean(list(per_class_accuracy.values()))
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        metrics['precision_per_class'] = {
            self.class_names[i]: precision[i] for i in range(len(precision))
        }
        metrics['recall_per_class'] = {
            self.class_names[i]: recall[i] for i in range(len(recall))
        }
        metrics['f1_per_class'] = {
            self.class_names[i]: f1[i] for i in range(len(f1))
        }
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        metrics['macro_precision'] = precision_macro
        metrics['macro_recall'] = recall_macro
        metrics['macro_f1'] = f1_macro
        metrics['weighted_precision'] = precision_weighted
        metrics['weighted_recall'] = recall_weighted
        metrics['weighted_f1'] = f1_weighted
        
        # Top-k accuracy (if probabilities available)
        if self.all_probabilities:
            probabilities = np.array(self.all_probabilities)
            for k in [3, 5]:
                if k <= self.num_classes:
                    metrics[f'top{k}_accuracy'] = top_k_accuracy_score(
                        targets, probabilities, k=k
                    )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
        
        return metrics
    
    def _calculate_per_class_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy for each class individually."""
        per_class_acc = {}
        
        for i, class_name in enumerate(self.class_names):
            # Find samples belonging to this class
            class_mask = targets == i
            if np.sum(class_mask) > 0:  # Only if there are samples for this class
                class_predictions = predictions[class_mask]
                class_targets = targets[class_mask]
                accuracy = accuracy_score(class_targets, class_predictions)
                per_class_acc[class_name] = accuracy
            else:
                per_class_acc[class_name] = 0.0
        
        return per_class_acc
    
    def print_metrics(self, metrics: Dict = None):
        """Print formatted metrics."""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        # Main metrics
        print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"Average Accuracy per Class: {metrics['average_accuracy_per_class']:.4f}")
        
        if 'top3_accuracy' in metrics:
            print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        if 'top5_accuracy' in metrics:
            print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        
        print(f"\nMacro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
        
        print(f"\nWeighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
        print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
        
        # Per-class accuracy
        print(f"\nPer-Class Accuracy:")
        print("-" * 40)
        for class_name, accuracy in metrics['per_class_accuracy'].items():
            print(f"{class_name:25s}: {accuracy:.4f}")
        
        print("="*60)
    
    def plot_confusion_matrix(self, metrics: Dict = None, save_path: str = None):
        """Plot confusion matrix."""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, metrics: Dict = None, save_path: str = None):
        """Plot per-class metrics."""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        classes = list(metrics['per_class_accuracy'].keys())
        accuracy = [metrics['per_class_accuracy'][cls] for cls in classes]
        precision = [metrics['precision_per_class'][cls] for cls in classes]
        recall = [metrics['recall_per_class'][cls] for cls in classes]
        f1 = [metrics['f1_per_class'][cls] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, metrics: Dict = None) -> str:
        """Generate detailed classification report."""
        if metrics is None:
            metrics = self.calculate_metrics()
        
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        
        report = classification_report(targets, predictions, 
                                     target_names=self.class_names,
                                     zero_division=0)
        return report


def evaluate_model(model, data_loader, device, class_names: List[str], 
                   criterion=None, plot_results: bool = True, 
                   save_plots: bool = False, save_dir: str = "./") -> Dict:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        class_names: List of class names
        criterion: Loss function (optional)
        plot_results: Whether to plot results
        save_plots: Whether to save plots
        save_dir: Directory to save plots
    
    Returns:
        Dictionary containing all metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator(class_names)
    
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics_calc.update(predictions, targets, probabilities)
            
            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_metrics()
    
    # Add loss if calculated
    if criterion and num_batches > 0:
        metrics['loss'] = total_loss / num_batches
    
    # Print metrics
    metrics_calc.print_metrics(metrics)
    
    # Generate detailed report
    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(metrics_calc.generate_classification_report(metrics))
    
    # Plot results
    if plot_results:
        if save_plots:
            cm_path = f"{save_dir}/confusion_matrix.png"
            metrics_path = f"{save_dir}/per_class_metrics.png"
        else:
            cm_path = None
            metrics_path = None
        
        metrics_calc.plot_confusion_matrix(metrics, cm_path)
        metrics_calc.plot_per_class_metrics(metrics, metrics_path)
    
    return metrics