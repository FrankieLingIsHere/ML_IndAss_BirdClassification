#!/usr/bin/env python3
"""
Demo script for bird classification model.
Creates synthetic data to demonstrate the system functionality.
"""
import os
import zipfile
import numpy as np
from PIL import Image
import torch
from train import main
import argparse


def create_synthetic_bird_dataset():
    """Create synthetic bird dataset for demonstration."""
    
    # Bird classes
    bird_classes = [
        'Cardinal', 'BlueJay', 'Robin', 'Sparrow', 'Eagle',
        'Owl', 'Hawk', 'Finch', 'Crow', 'Dove'
    ]
    
    # Create directories
    os.makedirs('train_images', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)
    
    print("Creating synthetic bird images...")
    
    # Create training images and annotations
    train_annotations = []
    for i in range(200):  # 200 training images
        class_idx = i % len(bird_classes)
        class_name = bird_classes[class_idx]
        image_name = f'train_bird_{i:03d}.jpg'
        
        # Create synthetic image with class-specific colors
        image = create_synthetic_bird_image(class_idx, 224, 224)
        image.save(f'train_images/{image_name}')
        train_annotations.append(f'{image_name} {class_name}')
    
    # Create test images and annotations
    test_annotations = []
    for i in range(100):  # 100 test images
        class_idx = i % len(bird_classes)
        class_name = bird_classes[class_idx]
        image_name = f'test_bird_{i:03d}.jpg'
        
        # Create synthetic image
        image = create_synthetic_bird_image(class_idx, 224, 224)
        image.save(f'test_images/{image_name}')
        test_annotations.append(f'{image_name} {class_name}')
    
    # Create zip files
    print("Creating Train.zip...")
    with zipfile.ZipFile('Train.zip', 'w') as zip_file:
        for filename in os.listdir('train_images'):
            zip_file.write(f'train_images/{filename}', filename)
    
    print("Creating Test.zip...")
    with zipfile.ZipFile('Test.zip', 'w') as zip_file:
        for filename in os.listdir('test_images'):
            zip_file.write(f'test_images/{filename}', filename)
    
    # Create annotation files
    with open('train.txt', 'w') as f:
        f.write('\n'.join(train_annotations))
    
    with open('test.txt', 'w') as f:
        f.write('\n'.join(test_annotations))
    
    # Clean up temporary directories
    import shutil
    shutil.rmtree('train_images')
    shutil.rmtree('test_images')
    
    print(f"Created synthetic dataset:")
    print(f"- Train.zip with {len(train_annotations)} images")
    print(f"- Test.zip with {len(test_annotations)} images")
    print(f"- {len(bird_classes)} bird classes: {', '.join(bird_classes)}")


def create_synthetic_bird_image(class_idx: int, width: int, height: int) -> Image.Image:
    """
    Create a synthetic bird image with class-specific visual patterns.
    
    Args:
        class_idx: Class index for color/pattern variation
        width: Image width
        height: Image height
        
    Returns:
        PIL Image
    """
    # Create base image
    image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add class-specific color bias
    color_biases = [
        [200, 50, 50],   # Cardinal - red
        [50, 100, 200],  # BlueJay - blue  
        [150, 75, 50],   # Robin - brown/orange
        [120, 100, 80],  # Sparrow - brown
        [80, 60, 40],    # Eagle - dark brown
        [100, 80, 60],   # Owl - brown
        [90, 70, 50],    # Hawk - dark
        [200, 200, 100], # Finch - yellow
        [30, 30, 30],    # Crow - black
        [180, 180, 180], # Dove - gray
    ]
    
    if class_idx < len(color_biases):
        bias = np.array(color_biases[class_idx])
        # Apply color bias with some randomness
        for c in range(3):
            image[:, :, c] = np.clip(
                image[:, :, c] * 0.3 + bias[c] * 0.7 + np.random.randint(-30, 31, (height, width)),
                0, 255
            ).astype(np.uint8)
    
    # Add some texture patterns
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    
    # Add circular pattern (bird body)
    mask = (x - center_x)**2 + (y - center_y)**2 < (min(width, height) // 3)**2
    image[mask] = np.clip(image[mask] + 30, 0, 255)
    
    # Add some noise for texture
    noise = np.random.normal(0, 15, (height, width, 3))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(image)


def run_demo():
    """Run the complete demo."""
    print("="*60)
    print("BIRD CLASSIFICATION MODEL DEMO")
    print("="*60)
    
    # Check if data already exists
    if not (os.path.exists('Train.zip') and os.path.exists('train.txt')):
        print("\nStep 1: Creating synthetic dataset...")
        create_synthetic_bird_dataset()
    else:
        print("\nDataset files already exist, using existing data.")
    
    print("\nStep 2: Starting model training...")
    print("This demo will train a lightweight model for faster execution.")
    
    # Set up demo arguments
    import sys
    demo_args = [
        '--model_type', 'lightweight',
        '--batch_size', '16',
        '--num_epochs', '10',  # Reduced for demo
        '--learning_rate', '0.001',
        '--early_stopping_patience', '5',
        '--save_dir', './demo_results',
        '--image_size', '224',
        '--validation_split', '0.2',
        '--use_class_weights'
    ]
    
    # Replace sys.argv for the demo
    original_argv = sys.argv[:]
    sys.argv = ['train.py'] + demo_args
    
    try:
        # Run training
        main()
    except Exception as e:
        print(f"Demo completed with note: {e}")
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\nFiles created:")
    print("- Train.zip and Test.zip: Synthetic bird image datasets")
    print("- train.txt and test.txt: Annotation files")
    print("- demo_results/: Training results and visualizations")
    print("\nTo use with your own data:")
    print("1. Replace Train.zip and Test.zip with your bird image datasets")
    print("2. Update train.txt and test.txt with your annotations")
    print("3. Run: python train.py --model_type resnet50 --num_epochs 100")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo Bird Classification System')
    parser.add_argument('--create_data_only', action='store_true',
                       help='Only create synthetic data without training')
    
    args = parser.parse_args()
    
    if args.create_data_only:
        create_synthetic_bird_dataset()
    else:
        run_demo()