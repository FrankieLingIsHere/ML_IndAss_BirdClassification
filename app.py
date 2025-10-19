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
import logging

# Import our model architecture
from models import create_model

# Optional: Hugging Face imports (used only when evaluating HF-format checkpoints)
try:
    import transformers
    from transformers import AutoConfig, AutoModelForImageClassification
    HF_AVAILABLE = True
except Exception:
    transformers = None
    HF_AVAILABLE = False

# HF image processor (AutoImageProcessor or AutoFeatureExtractor) will be stored here when available
hf_processor = None

# Configuration
# Default to the moved fine-tuned checkpoint if present
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join('results', 'fine_tune', 'best_model_finetuned.pth'))
# Optional: if your HF model id is known (e.g. Emiel/cub-200-bird-classifier-swin), set HF_MODEL_ID env var
HF_MODEL_ID = os.environ.get('HF_MODEL_ID', None)
CLASS_NAMES_PATH = os.environ.get('CLASS_NAMES_PATH', 'class_names.json')
FORCE_HF_LOAD = os.environ.get('FORCE_HF_LOAD', '0').lower() in ('1', 'true', 'yes')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default HF model id to try when checkpoint looks HF-like and HF_MODEL_ID not set
DEFAULT_HF_ID = 'Emiel/cub-200-bird-classifier-swin'

# Setup file logger for traceability in Spaces
LOG_FILE = os.environ.get('APP_LOG_PATH', 'app.log')
logging.basicConfig(level=logging.INFO, filename=LOG_FILE, filemode='a',
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load class names
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
    except Exception:
        class_names = []
else:
    class_names = []

NUM_CLASSES = len(class_names)


def load_checkpoint_model(model_path, device):
    """Attempt to load a checkpoint. Supports local create_model-based checkpoints and
    heuristic handling for Hugging Face (Swin) checkpoints when HF_MODEL_ID is set.
    Returns (model, actual_num_classes) or (None, None) on failure.
    """
    # Allow writing to module-level hf_processor
    global hf_processor

    # If user wants to force HF loading from hub, try that first (useful in Spaces)
    if FORCE_HF_LOAD and HF_MODEL_ID and HF_AVAILABLE:
        try:
            print(f"FORCE_HF_LOAD enabled: loading HF model from hub: {HF_MODEL_ID}")
            hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
            hf_model.to(device)
            hf_model.eval()
            # Try to load a matching image processor from the hub for preprocessing
            try:
                if transformers is not None:
                    hf_processor = transformers.AutoImageProcessor.from_pretrained(HF_MODEL_ID)
                    print("Loaded HF image processor from hub (force load)")
            except Exception:
                print("Warning: failed to load HF image processor for forced hub model")
            num_labels = getattr(hf_model.config, 'num_labels', NUM_CLASSES)
            print(f"Loaded HF model from hub with {num_labels} labels (force)")
            return hf_model, num_labels
        except Exception as e:
            print("Forced HF hub load failed:", e)

    if not os.path.exists(model_path):
        msg = f"Model file not found at {model_path}"
        print(msg)
        logger.info(msg)
        # If HF_MODEL_ID is set and transformers are available, try to load from hub
        if HF_MODEL_ID and HF_AVAILABLE:
            try:
                print(f"Attempting to load model from Hugging Face Hub: {HF_MODEL_ID}")
                hf_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
                hf_model.to(device)
                hf_model.eval()
                # Try to load image processor for preprocessing
                try:
                    if transformers is not None:
                        hf_processor = transformers.AutoImageProcessor.from_pretrained(HF_MODEL_ID)
                        print("Loaded HF image processor from hub")
                except Exception:
                    print("Warning: failed to load HF image processor from hub")
                num_labels = getattr(hf_model.config, 'num_labels', NUM_CLASSES)
                print(f"Loaded HF model from hub with {num_labels} labels")
                return hf_model, num_labels
            except Exception as e:
                print("Failed to load HF model from hub:", e)
        return None, None

    print(f"Loading checkpoint from: {model_path}")
    logger.info(f"Loading checkpoint from: {model_path}")
    try:
        ckpt = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print("Failed to load checkpoint file:", e)
        logger.exception("Failed to load checkpoint file:")
        ckpt = {}
    # unwrap common dict wrapper (support both 'model_state_dict' and 'state_dict')
    state_dict = {}
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt and isinstance(ckpt['model_state_dict'], dict):
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state_dict = ckpt['state_dict']
        else:
            # fallback: ckpt may already be a state dict
            state_dict = ckpt

    # If the state_dict is a single-key wrapper (e.g., {'state_dict': {...}} or {'model': {...}}), unwrap one more level
    if isinstance(state_dict, dict) and len(state_dict) == 1:
        sole_val = next(iter(state_dict.values()))
        if isinstance(sole_val, dict):
            # adopt inner dict as state_dict if it looks like parameters
            inner_keys = list(sole_val.keys())[:8]
            # Heuristic: keys with '.' and numeric shapes indicate a param dict
            if any('.' in k for k in inner_keys):
                logger.info(f"Unwrapping single-key checkpoint wrapper, inner keys sample: {inner_keys}")
                state_dict = sole_val

    # Diagnostic: print a few state_dict keys so we can tell checkpoint format
    try:
        sample_keys = list(state_dict.keys())[:16]
        print("Checkpoint sample keys:", sample_keys)
        logger.info(f"Checkpoint sample keys: {sample_keys}")
    except Exception:
        print("No state_dict keys to sample")
        logger.info("No state_dict keys to sample")

    # Heuristic: detect HF-style Swin checkpoint by looking for keys that start with 'swin.'
    # Detect HF-like keys; strip common 'module.' prefix before checking
    def key_is_hf_like(k: str) -> bool:
        kk = k.replace('module.', '')
        kk = kk.lower()
        return kk.startswith('swin.') or 'swin.embeddings' in kk or 'swin.patch_embeddings' in kk

    hf_like = any(key_is_hf_like(k) for k in state_dict.keys()) if state_dict else False
    hf_msg = f"hf_like_checkpoint_detected={hf_like} HF_AVAILABLE={HF_AVAILABLE} HF_MODEL_ID={'set' if HF_MODEL_ID else 'not-set'}"
    print(hf_msg)
    logger.info(hf_msg)

    if hf_like and HF_AVAILABLE:
        # choose which HF id to use: env var or default
        hf_id_to_use = HF_MODEL_ID or DEFAULT_HF_ID
        if HF_MODEL_ID is None:
            info_msg = f"HF_MODEL_ID not set; using DEFAULT_HF_ID='{DEFAULT_HF_ID}' to attempt hub load"
            print(info_msg)
            logger.info(info_msg)

        try:
            msg = f"Attempting to load Hugging Face model '{hf_id_to_use}' and apply checkpoint weights..."
            print(msg)
            logger.info(msg)
            # prefer using the hub config to instantiate exact architecture
            config = AutoConfig.from_pretrained(hf_id_to_use)
            hf_model = AutoModelForImageClassification.from_config(config)
            # load weights non-strictly: match shapes
            missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
            hf_model.to(device)
            hf_model.eval()
            # Try to fetch image processor for this hf id so we can preprocess in predict
            try:
                if transformers is not None:
                    hf_processor = transformers.AutoImageProcessor.from_pretrained(hf_id_to_use)
                    print(f"Loaded HF image processor for {hf_id_to_use}")
            except Exception:
                print(f"Warning: failed to load HF image processor for {hf_id_to_use}")
            ok_msg = f"Loaded HF model with non-strict state_dict (missing {len(missing)} keys, unexpected {len(unexpected)} keys)"
            print(ok_msg)
            logger.info(ok_msg)
            num_labels = getattr(hf_model.config, 'num_labels', NUM_CLASSES)
            return hf_model, num_labels
        except Exception as e:
            print("HF load failed:", e)
            logger.exception("HF load failed")
            print("Falling back to local model loader...")
            logger.info("Falling back to local model loader")

    # Fallback: try to detect EfficientNet-like shapes and create local model
    # Determine actual num classes by inspecting a likely classifier weight key
    actual_classes = NUM_CLASSES
    for k, v in state_dict.items():
        if k.endswith('classifier.9.weight') or k.endswith('classifier.weight'):
            try:
                actual_classes = v.shape[0]
                break
            except Exception:
                pass

    # Heuristic to choose an EfficientNet variant based on conv head size
    model_type = 'efficientnet_b2'
    if state_dict:
        if 'backbone._conv_head.weight' in state_dict:
            try:
                conv_head_shape = state_dict['backbone._conv_head.weight'].shape
                if conv_head_shape[0] == 1536:
                    model_type = 'efficientnet_b3'
                elif conv_head_shape[0] == 1408:
                    model_type = 'efficientnet_b2'
                elif conv_head_shape[0] == 1280:
                    model_type = 'efficientnet_b1'
            except Exception:
                pass

    print(f"Creating local model {model_type} with {actual_classes} classes (fallback)")
    model = create_model(num_classes=actual_classes, model_type=model_type, pretrained=False, dropout_rate=0.3)
    # Try to load state dict
    try:
        # if ckpt was a dict without model_state_dict, attempt to load directly
        to_load = state_dict if state_dict else ckpt
        model.load_state_dict(to_load, strict=False)
        model.to(device)
        model.eval()
        print("✅ Local model loaded (non-strict).")
        return model, actual_classes
    except Exception as e:
        print("Failed to load local model:", e)
        return None, None


# Load model
print("Loading model...", MODEL_PATH)
model, actual_classes = load_checkpoint_model(MODEL_PATH, DEVICE)
if model is None:
    print("No model available. The app will still launch but predictions will fail.")
else:
    print(f"Model ready. Classes={actual_classes}")
    # If this is a Hugging Face model with id2label, prefer that mapping
    try:
        hf_config = getattr(model, 'config', None)
        if hf_config is not None:
            id2label = getattr(hf_config, 'id2label', None)
            if id2label:
                # id2label keys may be strings or ints
                # Build ordered class_names list by index
                max_idx = max(int(k) for k in id2label.keys())
                hf_class_names = [""] * (max_idx + 1)
                for k, v in id2label.items():
                    hf_class_names[int(k)] = v.replace(' ', '_') if isinstance(v, str) else str(v)
                # Filter out empty entries
                hf_class_names = [c for c in hf_class_names if c]
                if len(hf_class_names) > 0:
                    class_names = hf_class_names
                    NUM_CLASSES = len(class_names)
                    print(f"Using Hugging Face id2label mapping with {NUM_CLASSES} classes")
    except Exception as e:
        print("Warning: failed to extract id2label from HF model config:", e)

    # Warn if class_names.json doesn't match model classes
    if class_names and actual_classes and len(class_names) != actual_classes:
        print(f"Warning: class_names.json has {len(class_names)} entries but model expects {actual_classes} classes.")
        # If HF labels exist and match expected size, prefer them
        if len(class_names) < actual_classes:
            print("Note: consider updating class_names.json to match the model's label order or set HF_MODEL_ID to use id2label mapping.")

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
        
        # If an HF image processor was loaded alongside a HF model, prefer it for preprocessing
        if hf_processor is not None:
            try:
                # AutoImageProcessor expects PIL images or numpy arrays; return_tensors='pt' gives PyTorch tensors
                proc = hf_processor(images=image, return_tensors='pt')
                # Some processors return 'pixel_values', others return 'pixel_values' key
                if 'pixel_values' in proc:
                    input_tensor = proc['pixel_values'].to(DEVICE)
                else:
                    # Fall back to first tensor-like value
                    val = next(iter(proc.values()))
                    input_tensor = val.to(DEVICE)
            except Exception:
                logger.exception('HF processor failed; falling back to torchvision preprocessing')
                hf_local_fallback = True
            else:
                hf_local_fallback = False
        else:
            hf_local_fallback = True

        if hf_local_fallback:
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

            # Handle Hugging Face ModelOutput objects
            try:
                # HF ModelOutput may be dict-like with a 'logits' attribute
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs
            except Exception:
                logits = outputs


            # Ensure logits is a tensor
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(np.asarray(logits)).to(DEVICE)

            # Handle unexpected tensor shapes:
            # - if logits has spatial dims (e.g., 4D), average them
            # - if logits is 1D, unsqueeze batch dim
            try:
                if logits.dim() > 2:
                    # average over all dims after channel dim
                    reduce_dims = tuple(range(2, logits.dim()))
                    logits = logits.mean(dim=reduce_dims)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
            except Exception:
                # if shape ops fail, log and return safe error
                logger.exception('Failed to normalize logits shape')
                return {"Error": 0.0}

            # If single-logit output, treat as sigmoid probability
            if logits.size(1) == 1:
                probs = torch.sigmoid(logits)
                # return single-label prob mapped to first class or generic
                prob = float(probs[0, 0].item())
                label = class_names[0].replace('_', ' ') if class_names else 'Class_0'
                return {label: prob}

            probabilities = F.softmax(logits, dim=1)
            # Get top 5 predictions
            top5_prob, top5_indices = torch.topk(probabilities, min(5, probabilities.shape[1]), dim=1)

            # Format results
            results = {}
            for i in range(top5_indices.shape[1]):
                class_idx = int(top5_indices[0][i].item())
                prob = float(top5_prob[0][i].item())
                # Handle potential class index mismatch
                if class_idx < len(class_names):
                    class_name = class_names[class_idx].replace('_', ' ')
                else:
                    class_name = "Class_" + str(class_idx)
                results[class_name] = prob

        return results
        
    except Exception as e:
        # Log exception and return a numeric-friendly error response for Gradio
        logger.exception('Prediction failed')
        print('Prediction failed:', e)
        return {"Error": 0.0}

# Create Gradio interface
title = "🐦 Bird Species Classifier"
description = """
## Advanced Bird Classification Model

This model can classify **199 different bird species** using advanced deep learning techniques:

### Model Details:
- **Architecture**: Auto-detected EfficientNet (B2/B3) with enhanced regularization
- **Training Strategy**: Progressive training with advanced augmentation
- **Training Strategy**: Progressive training with advanced augmentation — a balanced mix of RandAugment/AutoAugment schedules, probabilistic MixUp/CutMix and mild geometric/color jitter used to improve robustness while preserving fine-grained labels
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
- **Training**: Progressive training with advanced augmentation strategies (RandAugment + MixUp/CutMix + mild crops/color jitter)
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