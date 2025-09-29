#!/usr/bin/env python3
"""
Examples of how to use the bird classification system.
"""
import os
import sys


def example_basic_training():
    """Example 1: Basic training with default parameters."""
    print("="*60)
    print("EXAMPLE 1: Basic Training")
    print("="*60)
    
    command = """
python train.py \\
    --train_zip Train.zip \\
    --train_txt train.txt \\
    --test_zip Test.zip \\
    --test_txt test.txt
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Use ResNet50 with pretrained weights")
    print("- Train for up to 100 epochs with early stopping")
    print("- Use default hyperparameters")
    print("- Save results to ./results/")
    

def example_lightweight_model():
    """Example 2: Training with lightweight model."""
    print("="*60)
    print("EXAMPLE 2: Lightweight Model Training")
    print("="*60)
    
    command = """
python train.py \\
    --model_type lightweight \\
    --batch_size 16 \\
    --num_epochs 50 \\
    --learning_rate 0.01 \\
    --save_dir ./lightweight_results
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Use a custom lightweight CNN")
    print("- Smaller batch size for memory efficiency")
    print("- Higher learning rate for faster convergence")
    print("- Train for 50 epochs maximum")


def example_fine_tuning():
    """Example 3: Fine-tuning with frozen backbone."""
    print("="*60)
    print("EXAMPLE 3: Fine-tuning with Frozen Backbone")
    print("="*60)
    
    command = """
python train.py \\
    --model_type resnet18 \\
    --freeze_backbone \\
    --learning_rate 0.01 \\
    --num_epochs 30 \\
    --batch_size 64 \\
    --dropout_rate 0.7 \\
    --save_dir ./finetuning_results
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Use ResNet18 as backbone")
    print("- Freeze backbone weights (only train classifier)")
    print("- Use higher learning rate since only classifier is training")
    print("- Higher dropout for regularization")
    print("- Larger batch size for stable gradients")


def example_class_imbalance():
    """Example 4: Handling class imbalance."""
    print("="*60)
    print("EXAMPLE 4: Handling Class Imbalance")
    print("="*60)
    
    command = """
python train.py \\
    --model_type efficientnet_b0 \\
    --use_class_weights \\
    --batch_size 32 \\
    --learning_rate 0.001 \\
    --weight_decay 1e-3 \\
    --scheduler_type plateau \\
    --early_stopping_patience 20 \\
    --save_dir ./balanced_results
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Use EfficientNet-B0 for efficiency")
    print("- Automatically calculate and use class weights")
    print("- Use ReduceLROnPlateau scheduler")
    print("- Higher patience for early stopping")
    print("- Moderate weight decay for regularization")


def example_overfitting_prevention():
    """Example 5: Heavy regularization for overfitting prevention."""
    print("="*60)
    print("EXAMPLE 5: Overfitting Prevention")
    print("="*60)
    
    command = """
python train.py \\
    --model_type resnet50 \\
    --dropout_rate 0.6 \\
    --weight_decay 1e-2 \\
    --gradient_clip_value 1.0 \\
    --scheduler_type cosine \\
    --early_stopping_patience 10 \\
    --validation_split 0.3 \\
    --save_dir ./regularized_results
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Use high dropout rate (0.6)")
    print("- Strong L2 regularization (1e-2)")
    print("- Gradient clipping to prevent exploding gradients")
    print("- Cosine annealing learning rate schedule")
    print("- Larger validation split (30%)")
    print("- Earlier stopping with patience of 10")


def example_evaluation_only():
    """Example 6: Evaluation only with pre-trained model."""
    print("="*60)
    print("EXAMPLE 6: Evaluation Only")
    print("="*60)
    
    command = """
python train.py \\
    --evaluate_only \\
    --checkpoint_path ./results/checkpoints/best_model.pth \\
    --save_dir ./evaluation_results
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Load a pre-trained model from checkpoint")
    print("- Skip training and only run evaluation")
    print("- Generate all evaluation metrics and plots")


def example_demo_with_synthetic_data():
    """Example 7: Demo with synthetic data."""
    print("="*60)
    print("EXAMPLE 7: Demo with Synthetic Data")  
    print("="*60)
    
    command = """
python demo.py
"""
    
    print("Command:")
    print(command)
    print("\nThis will:")
    print("- Create synthetic bird images and annotations")
    print("- Train a lightweight model for demonstration")
    print("- Show all features of the system")
    print("- Complete end-to-end example")


def show_usage_tips():
    """Show general usage tips."""
    print("="*60)
    print("USAGE TIPS")
    print("="*60)
    
    tips = [
        "1. Always check your data format first using the demo",
        "2. Start with a lightweight model for initial experiments", 
        "3. Use class weights if you have imbalanced classes",
        "4. Monitor training curves to detect overfitting",
        "5. Use early stopping to prevent overtraining",
        "6. Try different learning rate schedules",
        "7. Increase dropout and weight decay if overfitting",
        "8. Use gradient clipping for unstable training",
        "9. Larger batch sizes often give more stable training",
        "10. Save checkpoints regularly for long training runs"
    ]
    
    for tip in tips:
        print(f"   {tip}")
        
    print("\nHyperparameter Guidelines:")
    print("-" * 25)
    print("Small dataset (< 1000 images): Use pretrained + freeze_backbone")
    print("Medium dataset (1K-10K): Fine-tune pretrained model")  
    print("Large dataset (> 10K): Can train from scratch")
    print()
    print("Overfitting indicators:")
    print("- Validation loss increases while training loss decreases")
    print("- Large gap between training and validation accuracy")
    print("- Model performs much worse on test set than validation set")
    print()
    print("Solutions for overfitting:")
    print("- Increase dropout_rate (0.3 → 0.6)")
    print("- Increase weight_decay (1e-4 → 1e-2)")
    print("- Decrease learning_rate") 
    print("- Use early_stopping with lower patience")
    print("- Increase validation_split")
    print("- Add more training data or data augmentation")


def main():
    """Run all examples."""
    examples = [
        example_basic_training,
        example_lightweight_model,
        example_fine_tuning,
        example_class_imbalance,
        example_overfitting_prevention,
        example_evaluation_only,
        example_demo_with_synthetic_data,
        show_usage_tips
    ]
    
    for i, example in enumerate(examples):
        example()
        if i < len(examples) - 1:
            input("\nPress Enter to continue to next example...")
            print()


if __name__ == '__main__':
    main()