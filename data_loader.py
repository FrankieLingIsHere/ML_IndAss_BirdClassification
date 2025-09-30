"""
Data loading utilities for bird classification dataset.
Handles data/Train/ directory, data/Test/ directory and data/train.txt, data/test.txt format.
"""
import os
import zipfile
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


class BirdDataset(Dataset):
    """Dataset class for bird classification."""
    
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to the directory containing images (e.g., 'data/Train' or 'data/Test')
            annotation_file: Path to the annotation text file 
            transform: Image transforms to apply
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform
        
        # Load annotations
        self.samples, self.class_to_idx, self.classes = self._load_annotations()
        
    def _load_annotations(self) -> Tuple[List[Tuple[str, int]], Dict[str, int], List[str]]:
        """
        Load annotations from text file.
        Format: image_name class_label
        
        Returns:
            samples: List of (image_path, class_index) tuples
            class_to_idx: Dictionary mapping class names to indices
            classes: List of class names
        """
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")
        
        samples = []
        class_names = set()
        
        # Read annotation file
        with open(self.annotation_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        # Try to parse as numeric class index first
                        try:
                            class_label = int(parts[1])
                        except ValueError:
                            # If not numeric, treat as string class name
                            class_label = ' '.join(parts[1:])  # Handle multi-word class names
                        samples.append((image_name, class_label))
        
        # Check if we have numeric or string labels
        first_label = samples[0][1] if samples else None
        if isinstance(first_label, int):
            # Numeric labels - extract real class names from image filenames if possible
            class_idx_to_name = {}
            
            # Try to extract class names from training image filenames
            for image_name, class_idx in samples:
                if class_idx not in class_idx_to_name:
                    # Extract species name from filename (before the first underscore followed by digits)
                    # Example: "Black_footed_Albatross_0004_2731401028.jpg" -> "Black_footed_Albatross"
                    name_parts = image_name.replace('.jpg', '').split('_')
                    species_name = ''
                    for i, part in enumerate(name_parts):
                        if part.isdigit():
                            species_name = '_'.join(name_parts[:i])
                            break
                    
                    if species_name:
                        class_idx_to_name[class_idx] = species_name
                    else:
                        class_idx_to_name[class_idx] = f"class_{class_idx}"
            
            # Create ordered class names list
            max_class_idx = max(class_idx_to_name.keys()) if class_idx_to_name else 0
            classes = []
            for i in range(max_class_idx + 1):
                if i in class_idx_to_name:
                    classes.append(class_idx_to_name[i])
                else:
                    classes.append(f"class_{i}")
            
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # Convert samples to use consistent format (image_path, class_index)
            indexed_samples = []
            for image_name, class_idx in samples:
                image_path = self._find_image_path(image_name)
                if image_path:
                    indexed_samples.append((image_path, class_idx))
        else:
            # String labels - create class mapping as before
            class_names = set(sample[1] for sample in samples)
            classes = sorted(list(class_names))
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # Convert samples to use class indices
            indexed_samples = []
            for image_name, class_label in samples:
                image_path = self._find_image_path(image_name)
                if image_path:
                    indexed_samples.append((image_path, class_to_idx[class_label]))
        
        print(f"Loaded {len(indexed_samples)} samples with {len(classes)} classes")
        print(f"Classes: {classes}")
        
        return indexed_samples, class_to_idx, classes
    
    def _find_image_path(self, image_name: str) -> str:
        """Find the full path to an image file."""
        # Try different extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Check if image_name already has extension
        if any(image_name.lower().endswith(ext) for ext in extensions):
            possible_paths = [
                os.path.join(self.image_dir, image_name),
                # Check subdirectories
                *[os.path.join(root, image_name) 
                  for root, dirs, files in os.walk(self.image_dir) 
                  if image_name in files]
            ]
        else:
            # Try adding extensions
            possible_paths = []
            for ext in extensions:
                full_name = image_name + ext
                possible_paths.extend([
                    os.path.join(self.image_dir, full_name),
                    # Check subdirectories
                    *[os.path.join(root, full_name) 
                      for root, dirs, files in os.walk(self.image_dir) 
                      if full_name in files]
                ])
        
        # Return first existing path
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        print(f"Warning: Could not find image {image_name}")
        return ""
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        try:
            # Load image if path exists
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # Create a dummy image if path is not found
                dummy_image = Image.new('RGB', (224, 224), 0)  # Black image
                image = dummy_image
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image and label
            dummy_image = Image.new('RGB', (224, 224), 0)  # Black image
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label


def get_data_transforms(image_size: int = 224, is_training: bool = True, augmentation_level: str = 'basic'):
    """
    Get data transforms for training and validation.
    
    Args:
        image_size: Size to resize images to
        is_training: Whether this is for training (includes data augmentation)
        augmentation_level: Level of augmentation ('basic', 'advanced', 'heavy')
    
    Returns:
        Transform pipeline
    """
    if is_training:
        if augmentation_level == 'basic':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
        elif augmentation_level == 'advanced':
            transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
            ])
        elif augmentation_level == 'heavy':
            transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
                transforms.RandomGrayscale(p=0.15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3))
            ])
        else:
            raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    else:
        transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.05), int(image_size * 1.05))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(train_dir: str, train_txt: str, test_dir: str, test_txt: str, 
                       batch_size: int = 32, image_size: int = 224, num_workers: int = 4,
                       validation_split: float = 0.2, augmentation_level: str = 'advanced'):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_dir: Path to training images directory (e.g., 'data/Train')
        train_txt: Path to training annotations file
        test_dir: Path to test images directory (e.g., 'data/Test')
        test_txt: Path to test annotations file
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        num_workers: Number of workers for data loading
        validation_split: Fraction of training data to use for validation
        augmentation_level: Level of data augmentation ('basic', 'advanced', 'heavy')
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, class_names
    """
    # Create transforms
    train_transform = get_data_transforms(image_size, is_training=True, augmentation_level=augmentation_level)
    val_test_transform = get_data_transforms(image_size, is_training=False)
    
    # Create full training dataset
    full_train_dataset = BirdDataset(train_dir, train_txt, transform=train_transform)
    
    # Split training data into train and validation
    full_size = len(full_train_dataset)
    val_size = int(validation_split * full_size)
    train_size = full_size - val_size
    
    train_dataset, temp_val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    # Create separate validation dataset with different transforms
    val_dataset_with_transform = BirdDataset(train_dir, train_txt, transform=val_test_transform)
    # Get the same indices as temp_val_dataset but with validation transforms
    val_indices = temp_val_dataset.indices
    val_dataset = Subset(val_dataset_with_transform, val_indices)
    
    # Create test dataset
    test_dataset = BirdDataset(test_dir, test_txt, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    num_classes = len(full_train_dataset.classes)
    class_names = full_train_dataset.classes
    
    return train_loader, val_loader, test_loader, num_classes, class_names