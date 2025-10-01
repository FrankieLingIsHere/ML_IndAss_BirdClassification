import json
import matplotlib.pyplot as plt
import numpy as np

def load_training_history(filepath='results_stage2_accelerated/training_history.json'):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history

def create_training_plots(history, save_path='training_performance_plots.png'):
    """Create comprehensive training performance plots"""
    
    # Extract data - using correct key names from your JSON structure
    epochs = list(range(1, len(history) + 1))
    train_acc = [epoch_data['train_acc'] for epoch_data in history]
    val_acc = [epoch_data['val_acc'] for epoch_data in history]
    train_loss = [epoch_data['train_loss'] for epoch_data in history]
    val_loss = [epoch_data['val_loss'] for epoch_data in history]
    learning_rates = [epoch_data['lr'] for epoch_data in history]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EfficientNet-B3 Training Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Accuracy
    ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax1.axvline(x=27, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 27)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Accuracy Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Highlight best performance
    best_epoch_idx = 26  # Epoch 27 is index 26
    ax1.annotate('Peak: {:.2f}%'.format(val_acc[best_epoch_idx]), 
                xy=(27, val_acc[best_epoch_idx]), xytext=(22, val_acc[best_epoch_idx]+5),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red', fontweight='bold')
    
    # Plot 2: Training and Validation Loss
    ax2.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax2.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax2.axvline(x=27, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 27)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    ax3.plot(epochs, learning_rates, 'g-', linewidth=2, marker='d', markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Cosine Annealing Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 4: Train-Validation Gap Analysis
    gap = [train_acc[i] - val_acc[i] for i in range(len(epochs))]
    ax4.plot(epochs, gap, 'purple', linewidth=2, marker='v', markersize=4)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Target Gap (~10%)')
    ax4.axvline(x=27, color='green', linestyle='--', alpha=0.7, label='Best Model (Epoch 27)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Training - Validation Accuracy (%)')
    ax4.set_title('Overfitting Analysis (Train-Val Gap)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add performance metrics text box
    max_val_acc = max(val_acc)
    max_val_epoch = val_acc.index(max_val_acc) + 1
    total_time_minutes = sum([epoch_data['time'] for epoch_data in history])/60
    
    textstr = '''Final Performance Metrics:
    • Peak Validation Accuracy: {:.2f}% (Epoch {})
    • Final Train-Val Gap: {:.2f}%
    • Training Time: ~{:.1f} minutes
    • Test Accuracy: 79.73%'''.format(
        max_val_acc, max_val_epoch, gap[26], total_time_minutes)
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=10, verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Training plots saved to: {}".format(save_path))
    
    return save_path

def create_performance_distribution_plot(eval_results_path='final_evaluation_results.json'):
    """Create class performance distribution plot"""
    
    with open(eval_results_path, 'r') as f:
        results = json.load(f)
    
    # Extract per-class accuracies
    class_accuracies = [details['accuracy'] for details in results['per_class_details'].values()]
    
    # Create performance distribution plot
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(class_accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=float(np.mean(class_accuracies)), color='red', linestyle='--', linewidth=2, 
                label='Mean: {:.1f}%'.format(np.mean(class_accuracies)))
    plt.xlabel('Class Accuracy (%)')
    plt.ylabel('Number of Classes')
    plt.title('Distribution of Per-Class Accuracies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance ranges
    plt.subplot(1, 2, 2)
    ranges = ['Perfect\n(100%)', 'Excellent\n(80-99%)', 'Good\n(60-79%)', 'Challenging\n(<60%)']
    counts = [
        sum(1 for acc in class_accuracies if acc == 100),
        sum(1 for acc in class_accuracies if 80 <= acc < 100),
        sum(1 for acc in class_accuracies if 60 <= acc < 80),
        sum(1 for acc in class_accuracies if acc < 60)
    ]
    colors = ['green', 'lightgreen', 'orange', 'lightcoral']
    
    bars = plt.bar(ranges, counts, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Number of Classes')
    plt.title('Performance Categories')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_performance_distribution.png', dpi=300, bbox_inches='tight')
    print("Class performance distribution saved to: class_performance_distribution.png")

if __name__ == "__main__":
    # Load training history
    history = load_training_history()
    
    # Create training performance plots
    plot_path = create_training_plots(history)
    
    # Create class performance distribution
    create_performance_distribution_plot()
    
    print("\nAll visualization plots generated successfully!")
    print("Files created:")
    print("1. training_performance_plots.png - Training curves and analysis")
    print("2. class_performance_distribution.png - Test performance distribution")