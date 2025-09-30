"""
Mixup and CutMix augmentation for improved generalization.
"""
import torch
import torch.nn.functional as F
import numpy as np


class MixupCutmix:
    """
    Mixup and CutMix augmentation for training.
    """
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        """
        Initialize Mixup/CutMix augmentation.
        
        Args:
            mixup_alpha: Mixup interpolation strength
            cutmix_alpha: CutMix interpolation strength  
            prob: Probability of applying augmentation
            switch_prob: Probability of using CutMix vs Mixup
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        
    def __call__(self, batch, targets):
        """
        Apply mixup or cutmix to a batch.
        
        Args:
            batch: Input batch [B, C, H, W]
            targets: Target labels [B]
            
        Returns:
            Mixed batch, mixed targets
        """
        if np.random.rand() > self.prob:
            return batch, targets
            
        if np.random.rand() < self.switch_prob:
            return self.cutmix(batch, targets)
        else:
            return self.mixup(batch, targets)
    
    def mixup(self, batch, targets):
        """Apply mixup augmentation."""
        batch_size = batch.size(0)
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
            
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        # Mix images
        mixed_batch = lam * batch + (1 - lam) * batch[indices]
        
        # Mix targets
        targets_a, targets_b = targets, targets[indices]
        mixed_targets = {
            'targets_a': targets_a,
            'targets_b': targets_b,
            'lam': lam
        }
        
        return mixed_batch, mixed_targets
    
    def cutmix(self, batch, targets):
        """Apply CutMix augmentation."""
        batch_size = batch.size(0)
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1
            
        # Shuffle indices
        indices = torch.randperm(batch_size)
        
        # Generate random box
        _, _, H, W = batch.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        mixed_batch = batch.clone()
        mixed_batch[:, :, bby1:bby2, bbx1:bbx2] = batch[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Mix targets
        targets_a, targets_b = targets, targets[indices]
        mixed_targets = {
            'targets_a': targets_a,
            'targets_b': targets_b,
            'lam': lam
        }
        
        return mixed_batch, mixed_targets


def mixup_loss_fn(criterion, outputs, mixed_targets):
    """
    Calculate loss for mixup/cutmix augmented data.
    
    Args:
        criterion: Loss function
        outputs: Model outputs
        mixed_targets: Mixed target dictionary
        
    Returns:
        Mixed loss
    """
    if isinstance(mixed_targets, dict):
        targets_a = mixed_targets['targets_a']
        targets_b = mixed_targets['targets_b']
        lam = mixed_targets['lam']
        
        loss_a = criterion(outputs, targets_a)
        loss_b = criterion(outputs, targets_b)
        
        return lam * loss_a + (1 - lam) * loss_b
    else:
        # Regular targets
        return criterion(outputs, mixed_targets)


class TestTimeAugmentation:
    """
    Test Time Augmentation for improved inference accuracy.
    """
    
    def __init__(self, n_tta=5, image_size=288):
        """
        Initialize TTA.
        
        Args:
            n_tta: Number of TTA iterations
            image_size: Image size for augmentation
        """
        self.n_tta = n_tta
        self.image_size = image_size
        
        # TTA transforms
        self.tta_transforms = [
            # Original
            torch.nn.Identity(),
            # Horizontal flip
            torch.nn.Sequential(
                torch.nn.ReflectionPad2d(0),
                torch.jit.script(lambda x: torch.flip(x, dims=[-1]))
            ),
            # Slight rotations and crops
            *[self._get_random_transform() for _ in range(n_tta - 2)]
        ]
    
    def _get_random_transform(self):
        """Get a random augmentation transform."""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(self.image_size, scale=(0.95, 1.0)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
    
    def __call__(self, model, inputs, device):
        """
        Apply TTA to model predictions.
        
        Args:
            model: Trained model
            inputs: Input batch
            device: Device to run on
            
        Returns:
            Averaged predictions
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(self.n_tta):
                # Apply augmentation
                if i == 0:
                    # Original image
                    augmented_inputs = inputs
                elif i == 1:
                    # Horizontal flip
                    augmented_inputs = torch.flip(inputs, dims=[-1])
                else:
                    # Random augmentations (simplified for inference)
                    augmented_inputs = inputs
                    # Add slight noise for variation
                    noise = torch.randn_like(inputs) * 0.01
                    augmented_inputs = augmented_inputs + noise
                
                # Get predictions
                outputs = model(augmented_inputs)
                probabilities = F.softmax(outputs, dim=1)
                predictions.append(probabilities)
        
        # Average predictions
        avg_predictions = torch.stack(predictions).mean(dim=0)
        return avg_predictions