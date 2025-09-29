"""
Data loading utilities for bird classification dataset.
Handles Train.zip/Test.zip and train.txt/test.txt format.
"""
import os
import zipfile
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class BirdDataset(Dataset):
    """Dataset class for bird classification."""
    
    def __init__(self, zip_path: str, annotation_file: str, transform=None, extract_dir: str = None):
        """
        Initialize the dataset.
        
        Args:
            zip_path: Path to the zip file containing images
            annotation_file: Path to the annotation text file 
            transform: Image transforms to apply
            extract_dir: Directory to extract images to (if None, uses temp directory)
        """
        self.zip_path = zip_path
        self.annotation_file = annotation_file
        self.transform = transform
        
        # Set extraction directory
        if extract_dir is None:
            self.extract_dir = os.path.join(os.path.dirname(zip_path), 
                                          f"extracted_{os.path.basename(zip_path).replace('.zip', '')}")
        else:
            self.extract_dir = extract_dir
        
        # Extract images if not already extracted
        self._extract_images()
        
        # Load annotations
        self.samples, self.class_to_idx, self.classes = self._load_annotations()
        
    def _extract_images(self):
        """Extract images from zip file if not already extracted."""
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir, exist_ok=True)
            print(f"Extracting {self.zip_path} to {self.extract_dir}")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
    
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
                        class_label = ' '.join(parts[1:])  # Handle multi-word class names
                        class_names.add(class_label)
                        samples.append((image_name, class_label))
        
        # Create class mappings
        classes = sorted(list(class_names))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Convert samples to use class indices
        indexed_samples = []
        for image_name, class_label in samples:
            # Find the actual image path
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
                os.path.join(self.extract_dir, image_name),
                # Check subdirectories
                *[os.path.join(root, image_name) 
                  for root, dirs, files in os.walk(self.extract_dir) 
                  if image_name in files]
            ]
        else:
            # Try adding extensions
            possible_paths = []
            for ext in extensions:
                full_name = image_name + ext
                possible_paths.extend([
                    os.path.join(self.extract_dir, full_name),
                    # Check subdirectories
                    *[os.path.join(root, full_name) 
                      for root, dirs, files in os.walk(self.extract_dir) 
                      if full_name in files]
                ])
        
        # Return first existing path
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        print(f"Warning: Could not find image {image_name}")
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image and label
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label


def get_data_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get data transforms for training and validation.
    
    Args:
        image_size: Size to resize images to
        is_training: Whether this is for training (includes data augmentation)
    
    Returns:
        Transform pipeline
    """
    if is_training:
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
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(train_zip: str, train_txt: str, test_zip: str, test_txt: str, 
                       batch_size: int = 32, image_size: int = 224, num_workers: int = 4,
                       validation_split: float = 0.2):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_zip: Path to training images zip file
        train_txt: Path to training annotations file
        test_zip: Path to test images zip file  
        test_txt: Path to test annotations file
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        num_workers: Number of workers for data loading
        validation_split: Fraction of training data to use for validation
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, class_names
    """
    # Create transforms
    train_transform = get_data_transforms(image_size, is_training=True)
    val_test_transform = get_data_transforms(image_size, is_training=False)
    
    # Create full training dataset
    full_train_dataset = BirdDataset(train_zip, train_txt, transform=train_transform)
    
    # Split training data into train and validation
    full_size = len(full_train_dataset)
    val_size = int(validation_split * full_size)
    train_size = full_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_test_transform
    
    # Create test dataset
    test_dataset = BirdDataset(test_zip, test_txt, transform=val_test_transform)
    
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