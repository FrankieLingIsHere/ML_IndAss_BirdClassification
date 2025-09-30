"""
Bird classification model architectures with overfitting prevention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

# Try to import EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("EfficientNet not available. Install with: pip install efficientnet-pytorch")


class BirdClassifier(nn.Module):
    """
    Bird classification model with ResNet backbone and overfitting prevention.
    """
    
    def __init__(self, num_classes: int, architecture: str = 'resnet50', 
                 pretrained: bool = True, dropout_rate: float = 0.5,
                 freeze_backbone: bool = False):
        """
        Initialize the bird classifier.
        
        Args:
            num_classes: Number of bird classes
            architecture: Backbone architecture ('resnet50', 'resnet18', 'efficientnet_b0')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone weights
        """
        super(BirdClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Choose backbone architecture
        if architecture == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
        elif architecture == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif architecture in ['efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4'] and EFFICIENTNET_AVAILABLE:
            model_name = architecture.replace('_', '-')
            if pretrained:
                self.backbone = EfficientNet.from_pretrained(model_name)
            else:
                self.backbone = EfficientNet.from_name(model_name)
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Enhanced classifier head with batch normalization and progressive dimension reduction
        # Reduced complexity to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(p=min(dropout_rate * 1.5, 0.7)),  # Increased dropout
            nn.Linear(num_features, 512),  # Reduced from 1024
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=min(dropout_rate * 1.2, 0.6)),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class LightweightBirdClassifier(nn.Module):
    """
    Lightweight CNN model for bird classification with batch normalization.
    """
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        """
        Initialize lightweight classifier.
        
        Args:
            num_classes: Number of bird classes
            dropout_rate: Dropout rate for regularization
        """
        super(LightweightBirdClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout_rate/2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(num_classes: int, model_type: str = 'resnet50', 
                pretrained: bool = True, dropout_rate: float = 0.5,
                freeze_backbone: bool = False) -> nn.Module:
    """
    Create a bird classification model.
    
    Args:
        num_classes: Number of bird classes
        model_type: Type of model ('resnet50', 'resnet18', 'efficientnet_b0', 'lightweight')
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout rate for regularization
        freeze_backbone: Whether to freeze backbone weights (ignored for lightweight model)
    
    Returns:
        PyTorch model
    """
    if model_type == 'lightweight':
        return LightweightBirdClassifier(num_classes, dropout_rate)
    else:
        return BirdClassifier(num_classes, model_type, pretrained, 
                             dropout_rate, freeze_backbone)


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models for improved performance.
    """
    
    def __init__(self, models_list: list):
        """
        Initialize model ensemble.
        
        Args:
            models_list: List of trained models to ensemble
        """
        super(ModelEnsemble, self).__init__()
        self.models = nn.ModuleList(models_list)
        
    def forward(self, x):
        """Forward pass through all models and average predictions."""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        return torch.log(ensemble_pred + 1e-8)  # Convert back to log probabilities