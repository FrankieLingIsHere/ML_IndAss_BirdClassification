#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start script for enhanced bird classification training.
"""
import subprocess
import sys
import os

def install_requirements():
    """Install enhanced requirements."""
    print("Installing enhanced requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "efficientnet-pytorch"])
        print("EfficientNet installed successfully!")
    except subprocess.CalledProcessError:
        print("Could not install EfficientNet. Will use ResNet models only.")

def run_enhanced_training():
    """Run enhanced training with optimal settings."""
    print("Starting enhanced bird classification training...")
    
    # Configuration for best performance
    cmd = [
        sys.executable, "train_enhanced.py",
        "--model_type", "efficientnet_b3",
        "--batch_size", "24",
        "--num_epochs", "60", 
        "--learning_rate", "0.0001",
        "--loss_type", "focal",
        "--focal_gamma", "2.0",
        "--augmentation_level", "advanced",
        "--image_size", "288",
        "--dropout_rate", "0.3",
        "--scheduler_type", "cosine",
        "--early_stopping_patience", "15",
        "--save_dir", "./results_enhanced"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("Enhanced training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print("Trying fallback with ResNet101...")
        
        # Fallback to ResNet101
        cmd_fallback = [
            sys.executable, "train_enhanced.py",
            "--model_type", "resnet101",
            "--batch_size", "24",
            "--num_epochs", "60",
            "--learning_rate", "0.0001", 
            "--loss_type", "focal",
            "--augmentation_level", "advanced",
            "--image_size", "288",
            "--save_dir", "./results_enhanced_resnet101"
        ]
        
        try:
            subprocess.run(cmd_fallback, check=True)
            print("Fallback training completed successfully!")
        except subprocess.CalledProcessError as e2:
            print(f"Fallback training also failed: {e2}")

def run_multistage_training():
    """Run multi-stage training for maximum performance."""
    print("Starting multi-stage enhanced training...")
    
    cmd = [
        sys.executable, "train_enhanced.py",
        "--model_type", "efficientnet_b3",
        "--multistage_training",
        "--stage1_epochs", "15",
        "--stage2_epochs", "30", 
        "--stage3_epochs", "15",
        "--batch_size", "24",
        "--learning_rate", "0.0001",
        "--loss_type", "focal",
        "--augmentation_level", "advanced",
        "--image_size", "288",
        "--save_dir", "./results_multistage"
    ]
    
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        print("Multi-stage training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Multi-stage training failed: {e}")

def main():
    """Main function."""
    print("Enhanced Bird Classification Training")
    print("===================================")
    
    # Check if data exists
    if not os.path.exists("data/Train") or not os.path.exists("data/train.txt"):
        print("Error: Dataset not found!")
        print("Please ensure data/Train directory and data/train.txt exist.")
        return
    
    print("1. Install enhanced requirements")
    install_requirements()
    
    print("\n2. Choose training strategy:")
    print("   a) Enhanced single-stage training (recommended)")
    print("   b) Multi-stage training (maximum performance)")
    print("   c) Run both")
    
    choice = input("\nEnter your choice (a/b/c): ").lower().strip()
    
    if choice == 'a':
        run_enhanced_training()
    elif choice == 'b':
        run_multistage_training()
    elif choice == 'c':
        run_enhanced_training()
        print("\n" + "="*50)
        run_multistage_training()
    else:
        print("Invalid choice. Running enhanced single-stage training...")
        run_enhanced_training()
    
    print("\nTraining process completed!")
    print("Check the results directories for outputs.")

if __name__ == "__main__":
    main()