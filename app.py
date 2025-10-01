"""
Gradio App for Bird Classification - Hugging Face Deployment
Enhanced model with architecture auto-detection and error handling.
"""
import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import json
import numpy as np
from torchvision import transforms
import os

# Import our model architecture
from models import create_model

# Configuration
MODEL_PATH = "best_model.pth"
CLASS_NAMES_PATH = "class_names.json"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# Load model - detect architecture from checkpoint
print("Loading model...")

# First, try to detect the correct architecture from the model file
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Detect EfficientNet variant based on feature dimensions
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check backbone head feature size to determine EfficientNet variant
    if 'backbone._conv_head.weight' in state_dict:
        conv_head_shape = state_dict['backbone._conv_head.weight'].shape
        if conv_head_shape[0] == 1536:  # EfficientNet-B3
            model_type = 'efficientnet_b3'
        elif conv_head_shape[0] == 1408:  # EfficientNet-B2
            model_type = 'efficientnet_b2'
        elif conv_head_shape[0] == 1280:  # EfficientNet-B0/B1
            model_type = 'efficientnet_b1'
        else:
            model_type = 'efficientnet_b2'  # Default fallback
    else:
        model_type = 'efficientnet_b2'  # Default fallback
    
    # Check actual number of classes from classifier
    if 'classifier.9.weight' in state_dict:
        actual_classes = state_dict['classifier.9.weight'].shape[0]
    else:
        actual_classes = NUM_CLASSES
    
    print("Detected model: {} with {} classes".format(model_type, actual_classes))
    
else:
    model_type = 'efficientnet_b2'
    actual_classes = NUM_CLASSES
    print("Model file not found, using default: {}".format(model_type))

model = create_model(
    num_classes=actual_classes,
    model_type=model_type,
    pretrained=False,  # We're loading trained weights
    dropout_rate=0.3
)

# Load trained weights
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded successfully! ({}, {} classes)".format(model_type, actual_classes))
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded successfully! ({}, {} classes)".format(model_type, actual_classes))
    except Exception as e:
        print("❌ Error loading model: {}".format(str(e)))
        print("Please ensure the model architecture matches the saved weights.")
else:
    print("⚠️ Model file not found. Please ensure best_model.pth is in the repository.")

model.to(DEVICE)
model.eval()

def predict_bird(image):
    """
    Predict bird species from uploaded image.
    """
    try:
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define preprocessing step by step to avoid namespace issues
        resize = transforms.Resize((320, 320))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Apply transformations step by step
        resized_image = resize(image)
        tensor_image = to_tensor(resized_image)
        normalized_tensor = normalize(tensor_image)
        input_tensor = normalized_tensor.unsqueeze(0).to(DEVICE)
        
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, 5)
            
            # Format results
            results = {}
            for i in range(5):
                class_idx = top5_indices[0][i].item()
                prob = top5_prob[0][i].item()
                # Handle potential class index mismatch
                if class_idx < len(class_names):
                    class_name = class_names[class_idx].replace('_', ' ')
                else:
                    class_name = "Class_" + str(class_idx)
                results[class_name] = float(prob)
        
        return results
        
    except Exception as e:
        return {"Error": "Prediction failed: " + str(e)}

# Create Gradio interface
title = "🐦 Bird Species Classifier"
description = """
## Advanced Bird Classification Model

This model can classify **199 different bird species** using advanced deep learning techniques:

### Model Details:
- **Architecture**: Auto-detected EfficientNet (B2/B3) with enhanced regularization
- **Training Strategy**: Progressive training with advanced augmentation
- **Performance**: Optimized for accuracy and reliability
- **Dataset**: CUB-200-2011 (199 bird species)

### How to use:
1. Upload a clear image of a bird
2. The model will predict the top 5 most likely species
3. Confidence scores show the model's certainty

### Best Results Tips:
- Use high-quality, well-lit images
- Ensure the bird is clearly visible
- Close-up shots work better than distant ones
- Natural lighting produces better results

**Note**: This model was trained on the CUB-200-2011 dataset and works best with North American bird species.
"""

article = """
### Technical Implementation:
- **Framework**: PyTorch with auto-detected EfficientNet backbone
- **Training**: Progressive training with advanced augmentation strategies
- **Regularization**: Optimized dropout rates and comprehensive validation
- **Image Size**: 320x320 pixels for optimal detail capture

### About the Model:
This bird classifier was developed using advanced machine learning techniques including:
- Transfer learning from ImageNet-pretrained EfficientNet
- Progressive training strategy across multiple stages
- Advanced data augmentation for improved generalization
- Comprehensive evaluation and optimization

The model automatically detects the correct architecture (EfficientNet-B2 or B3) from the saved weights,
ensuring compatibility and optimal performance.

For more details about the training process and methodology, please refer to the repository documentation.
"""

# Create the interface
iface = gr.Interface(
    fn=predict_bird,
    inputs=gr.Image(type="pil", label="Upload Bird Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title=title,
    description=description,
    article=article,
    examples=[
        # You can add example images here if you have them
    ],
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch(debug=True)