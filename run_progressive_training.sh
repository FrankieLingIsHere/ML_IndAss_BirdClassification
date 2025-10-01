#!/bin/bash
# Terminal Commands for Progressive Training with TTA & MixUp
# From 75.50% to 90% accuracy target

echo "🚀 Progressive Bird Classification Training Pipeline"
echo "Current: 75.50% → Target: 90%"
echo "================================================================"

# Method 1: Quick MixUp Training (Recommended for immediate improvement)
echo "📋 METHOD 1: Enhanced MixUp Training"
echo "Expected: 75.50% → 80-82%"
echo "Command:"
echo "python train_mixup_progressive.py"
echo ""

# Method 2: Manual Progressive Stages
echo "📋 METHOD 2: Manual Progressive Training"
echo ""

echo "Stage 1: Enhanced Training with MixUp (320px images)"
echo "Expected: 80-82% accuracy"
echo "python -c \"
import torch
import sys
sys.path.append('.')
from train_mixup_progressive import train_with_mixup_progressive
train_with_mixup_progressive()
\""
echo ""

echo "Stage 2: EfficientNet-B3 Training (352px)"
echo "Expected: 82-85% accuracy"
echo "python train_enhanced_90.py --model_type efficientnet_b3 --image_size 352 --batch_size 12 --learning_rate 5e-5 --epochs 30 --save_dir ./results_b3_352"
echo ""

echo "Stage 3: Large Image Training (384px)"
echo "Expected: 85-87% accuracy"
echo "python train_enhanced_90.py --model_type efficientnet_b3 --image_size 384 --batch_size 8 --learning_rate 3e-5 --epochs 25 --save_dir ./results_b3_384"
echo ""

echo "Stage 4: Test Time Augmentation"
echo "Expected: 87-92% accuracy"
echo "python -c \"
import torch
from mixup import TestTimeAugmentation
from data_loader import create_data_loaders
from models import create_model
from metrics import evaluate_model
import torch.nn as nn

# Load best model
model = torch.load('./results_b3_384/best_model.pth', map_location='cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Load test data
_, _, test_loader, num_classes, class_names = create_data_loaders(
    train_dir='./data/Train', train_txt='./data/train.txt',
    test_dir='./data/Test', test_txt='./data/test.txt',
    batch_size=8, image_size=384, validation_split=0.2
)

# Apply TTA
tta = TestTimeAugmentation(n_tta=10, image_size=384)
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        tta_outputs = tta(model, inputs, device)
        _, predicted = tta_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

accuracy = correct / total
print(f'TTA Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
\""
echo ""

# Method 3: All-in-One Command
echo "📋 METHOD 3: All-in-One Progressive Pipeline"
echo "Expected: 75.50% → 90%+"
echo "python -c \"
exec(open('train_progressive_all.py').read())
\""
echo ""

echo "================================================================"
echo "💡 RECOMMENDED QUICK START:"
echo "1. Run: python train_mixup_progressive.py"
echo "2. Wait for completion (~2-3 hours)"
echo "3. Check accuracy improvement"
echo "4. If >82%, continue with progressive stages"
echo "================================================================"