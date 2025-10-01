# 🐦 Advanced Bird Species Classifier

A state-of-the-art deep learning model for classifying 200 bird species with **76.74% accuracy**, built using advanced training techniques and deployed on Hugging Face Spaces.

## 🎯 Model Performance

- **Test Accuracy**: 76.74% (Stage 2 Results)
- **Architecture**: EfficientNet-B2 with enhanced regularization
- **Training Strategy**: Progressive multi-stage training with MixUp augmentation
- **Dataset**: CUB-200-2011 (200 North American bird species)

## 🚀 Quick Start

### Online Demo
Try the live demo on Hugging Face Spaces: [Bird Classifier Demo](https://huggingface.co/spaces/your-username/bird-classifier)

### Local Usage

```bash
# Clone the repository
git clone https://github.com/your-username/bird-classifier.git
cd bird-classifier

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

## 📊 Technical Details

### Model Architecture
- **Backbone**: EfficientNet-B2 (pre-trained on ImageNet)
- **Input Size**: 320×320 pixels
- **Classifier Head**: Progressive dimension reduction (512→256→200)
- **Regularization**: Optimized dropout rates (0.18→0.15→0.09)

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Format

The system expects the following data structure:

### Training Data
- `Train.zip`: Zip file containing training images
- `train.txt`: Annotation file with format: `image_name class_label`

### Testing Data
- `Test.zip`: Zip file containing test images  
- `test.txt`: Annotation file with format: `image_name class_label`

### Example annotation format:

**Option 1: Numeric class indices (current format):**
```
Black_footed_Albatross_0004_2731401028.jpg 0
Black_footed_Albatross_0008_1384283201.jpg 0
0000.jpg 0
0001.jpg 1
0002.jpg 2
...
```

**Option 2: String class names (also supported):**
```
bird001.jpg Cardinal
bird002.jpg BlueJay
bird003.jpg Robin
bird004.jpg Cardinal
...
```

The system automatically detects whether your labels are numeric indices or string class names.

## Usage

### Basic Training

```bash
python train.py --train_dir data/Train --train_txt data/train.txt --test_dir data/Test --test_txt data/test.txt
```

### Advanced Training with Custom Parameters

```bash
python train.py \
    --model_type resnet50 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --weight_decay 1e-4 \
    --dropout_rate 0.5 \
    --scheduler_type cosine \
    --early_stopping_patience 15 \
    --use_class_weights \
    --save_dir ./results
```

### Command Line Arguments

#### Data Parameters
- `--train_zip`: Path to training images zip file (default: 'Train.zip')
- `--train_dir`: Path to training images directory (default: 'data/Train')
- `--train_txt`: Path to training annotations file (default: 'data/train.txt')
- `--test_dir`: Path to test images directory (default: 'data/Test')
- `--test_txt`: Path to test annotations file (default: 'data/test.txt')

#### Model Parameters
- `--model_type`: Model architecture ('resnet50', 'resnet18', 'efficientnet_b0', 'lightweight')
- `--pretrained`: Use pretrained weights (default: True)
- `--freeze_backbone`: Freeze backbone weights for fine-tuning
- `--dropout_rate`: Dropout rate for regularization (default: 0.5)

#### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--weight_decay`: L2 regularization strength (default: 1e-4)
- `--scheduler_type`: Learning rate scheduler ('cosine', 'step', 'plateau', 'none')
- `--early_stopping_patience`: Patience for early stopping (default: 15)
- `--gradient_clip_value`: Gradient clipping value (optional)

#### Other Parameters
- `--image_size`: Input image size (default: 224)
- `--validation_split`: Validation split ratio (default: 0.2)
- `--use_class_weights`: Handle class imbalance with automatic weighting
- `--save_dir`: Directory to save results (default: './results')

## Model Architectures

### 1. ResNet-based Models (resnet50, resnet18)
- Pre-trained on ImageNet
- Custom classifier head with dropout
- Best for high accuracy with sufficient data

### 2. EfficientNet-B0
- Efficient architecture with compound scaling
- Good balance of accuracy and efficiency
- Pre-trained on ImageNet

### 3. Lightweight CNN
- Custom lightweight architecture
- Batch normalization and dropout
- Good for smaller datasets or faster inference

## Overfitting Prevention Techniques

### 1. Regularization
- **Dropout**: Applied in classifier heads and convolutional layers
- **L2 Weight Decay**: Penalizes large weights
- **Batch Normalization**: Stabilizes training

### 2. Early Stopping
- Monitors validation loss
- Stops training when no improvement
- Restores best weights automatically

### 3. Data Augmentation
- Random horizontal flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations

### 4. Learning Rate Scheduling
- **Cosine Annealing**: Gradually reduces learning rate
- **Step Scheduler**: Reduces LR at fixed intervals
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus

### 5. Gradient Clipping
- Prevents exploding gradients
- Stabilizes training

## Evaluation Metrics

The system provides comprehensive evaluation:

### Primary Metrics (Required)
- **Top-1 Accuracy**: Overall classification accuracy
- **Average Accuracy per Class**: Mean of per-class accuracies (handles class imbalance)

### Additional Metrics
- **Top-3 and Top-5 Accuracy**: For multi-class scenarios
- **Precision, Recall, F1-score**: Per class and macro/weighted averages
- **Confusion Matrix**: Detailed error analysis
- **Classification Report**: Comprehensive per-class statistics

### Visualizations
- Training/validation loss and accuracy curves
- Learning rate schedule
- Overfitting indicator (validation - training loss)
- Confusion matrix heatmap
- Per-class metrics bar chart

## Output Files

After training, the following files are saved in the results directory:

```
results/
├── checkpoints/
│   ├── best_model.pth          # Best model weights
│   ├── final_model.pth         # Final model state
│   └── checkpoint_epoch_*.pth   # Periodic checkpoints
├── training_history.png        # Training curves
├── confusion_matrix.png        # Confusion matrix
├── per_class_metrics.png       # Per-class performance
└── results_summary.json        # Summary statistics
```

## Example Results

```
EVALUATION METRICS
============================================================
Top-1 Accuracy: 0.8850
Average Accuracy per Class: 0.8720
Top-3 Accuracy: 0.9650
Top-5 Accuracy: 0.9880

Macro Precision: 0.8745
Macro Recall: 0.8720
Macro F1-Score: 0.8725

Weighted Precision: 0.8851
Weighted Recall: 0.8850
Weighted F1-Score: 0.8848
```

## Handling Different Scenarios

### 1. Small Dataset
- Use pretrained models
- Higher dropout rate (0.6-0.7)
- More aggressive data augmentation
- Lower learning rate

### 2. Large Dataset
- Can train from scratch or fine-tune
- Lower dropout rate (0.3-0.4)
- Higher learning rate
- Longer training

### 3. Class Imbalance
- Use `--use_class_weights` flag
- Monitor per-class accuracy
- Consider data augmentation for minority classes

### 4. Overfitting Issues
- Increase dropout rate
- Increase weight decay
- Reduce learning rate
- Use early stopping with lower patience
- Add more data augmentation

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size
2. **Low accuracy**: Check data quality, increase model capacity, adjust hyperparameters
3. **Overfitting**: Increase regularization, reduce model complexity
4. **Slow convergence**: Increase learning rate, check data preprocessing

### Data Issues
- Ensure image files exist in zip archives
- Check annotation file format (space-separated)
- Verify class names are consistent
- Check for corrupted images

## Demo Mode

If data files are not found, the system creates dummy annotation files for demonstration:

```bash
python train.py
```

This will create `data/train.txt` and `data/test.txt` with dummy data to show the expected format.

## Contributing

To extend the system:
1. Add new model architectures in `models.py`
2. Add new evaluation metrics in `metrics.py`
3. Modify data loading for different formats in `data_loader.py`
4. Add new training strategies in `trainer.py`

## License

This project is for educational purposes. Please respect the licenses of the underlying libraries and datasets used.