# PowerShell Commands for Progressive Training with TTA & MixUp
# From 75.50% to 90% accuracy target

Write-Host "🚀 Progressive Bird Classification Training Pipeline" -ForegroundColor Green
Write-Host "Current: 75.50% → Target: 90%" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

# Method 1: Quick MixUp Training (Recommended)
Write-Host "📋 METHOD 1: Enhanced MixUp Training" -ForegroundColor Magenta
Write-Host "Expected: 75.50% → 80-82%" -ForegroundColor Yellow
Write-Host "Command:" -ForegroundColor White
Write-Host "python train_mixup_progressive.py" -ForegroundColor Green
Write-Host ""

# Method 2: Step-by-Step Commands
Write-Host "📋 METHOD 2: Manual Step-by-Step Training" -ForegroundColor Magenta
Write-Host ""

Write-Host "Stage 1: Enhanced MixUp Training (320px)" -ForegroundColor Cyan
Write-Host "Expected: 80-82% accuracy" -ForegroundColor Yellow
Write-Host 'python -c "import torch; import sys; sys.path.append('"'"'.'"'"'); from train_mixup_progressive import train_with_mixup_progressive; train_with_mixup_progressive()"' -ForegroundColor Green
Write-Host ""

Write-Host "Stage 2: EfficientNet-B3 (352px)" -ForegroundColor Cyan  
Write-Host "Expected: 82-85% accuracy" -ForegroundColor Yellow
Write-Host "python train_enhanced_90.py" -ForegroundColor Green
Write-Host ""

Write-Host "Stage 3: TTA Evaluation" -ForegroundColor Cyan
Write-Host "Expected: 85-90% accuracy" -ForegroundColor Yellow
Write-Host 'python -c "
import torch
from mixup import TestTimeAugmentation
from data_loader import create_data_loaders
import torch.nn as nn

# Load model
model = torch.load('"'"'./results_stage1_mixup/best_model.pth'"'"', map_location='"'"'cpu'"'"')
device = '"'"'cuda'"'"' if torch.cuda.is_available() else '"'"'cpu'"'"'
model = model.to(device)

# Load test data  
_, _, test_loader, num_classes, class_names = create_data_loaders(
    train_dir='"'"'./data/Train'"'"', train_txt='"'"'./data/train.txt'"'"',
    test_dir='"'"'./data/Test'"'"', test_txt='"'"'./data/test.txt'"'"',
    batch_size=8, image_size=320
)

# TTA
tta = TestTimeAugmentation(n_tta=8, image_size=320)
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
print(f'"'"'TTA Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)'"'"')
"' -ForegroundColor Green

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "💡 QUICK START COMMANDS:" -ForegroundColor Yellow
Write-Host "1. cd 'c:\Users\User\Documents\Degree Y3 S1\COS30082 Applied Machine Learning\Assignment(Individual)\ML_IndAss_BirdClassification'" -ForegroundColor White
Write-Host "2. python train_mixup_progressive.py" -ForegroundColor Green
Write-Host "3. Wait ~2-3 hours for training completion" -ForegroundColor Yellow
Write-Host "4. Check results in ./results_stage1_mixup/" -ForegroundColor White
Write-Host "================================================================" -ForegroundColor Cyan