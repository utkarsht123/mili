import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np

from modules.models import LightweightMultiModalDETR

# --- 1. Load Models ---

def load_object_detector(config_path="config.yaml"):
    """Loads your trained LightweightMultiModalDETR model."""
    print("Loading object detector...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = LightweightMultiModalDETR(config)
    model.load_state_dict(torch.load(config['demo']['model_path'], map_location=torch.device('cpu')))
    model.eval()
    print(" -> Object detector loaded.")
    return model, config

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """Loads a standard, compact CLIP model from OpenAI."""
    print("Loading zero-shot classifier (CLIP)...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(" -> CLIP model loaded.")
    return model, processor

# --- 2. The Main Inference Pipeline ---

def run_zero_shot_detection(detector, clip_model, clip_processor, image_path, text_prompts, detector_config):
    """
    Performs the two-stage detection and zero-shot classification.
    """
    original_image = Image.open(image_path).convert("RGB")
    
    # --- Stage 1: Object Detection ---
    preprocess = transforms.Compose([
        transforms.Resize(detector_config['data']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(original_image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = detector(img_tensor)
        
    probs = outputs['pred_logits'][0].softmax(-1)
    scores, _ = probs.max(-1)
    
    keep = scores > detector_config['demo']['confidence_threshold']
    detected_boxes = outputs['pred_boxes'][0][keep]

    # --- Stage 2: Zero-Shot Classification with CLIP ---
    final_labels = []
    final_boxes = []
    final_scores = []

    for box in detected_boxes:
        w, h = original_image.size
        x_c, y_c, w_b, h_b = box * torch.tensor([w, h, w, h])
        
        crop_box = [int(x_c - w_b/2), int(y_c - h_b/2), int(x_c + w_b/2), int(y_c + h_b/2)]
        
        cropped_image = original_image.crop(crop_box)
        
        inputs = clip_processor(text=text_prompts, images=cropped_image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        clip_probs = logits_per_image.softmax(dim=1)
        
        best_score, best_label_idx = clip_probs[0].max(dim=0)
        best_label = text_prompts[best_label_idx]
        
        final_labels.append(f"{best_label}")
        final_scores.append(best_score.item())
        final_boxes.append(box.numpy())

    return original_image, final_boxes, final_labels, final_scores

# --- 3. Visualization ---
def draw_results(image, boxes, labels, scores):
    img_draw = ImageDraw.Draw(image)
    colors = ['#FF3B30', '#FF9500', '#FFCC00', '#34C759', '#007AFF']
    for box, label, score in zip(boxes, labels, scores):
        w, h = image.size
        x_c, y_c, w_b, h_b = box * np.array([w, h, w, h])
        x1, y1, x2, y2 = x_c - w_b/2, y_c - h_b/2, x_c + w_b/2, y_c + h_b/2
        color = colors[hash(label) % len(colors)]
        img_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} ({score:.2f})"
        text_bbox = img_draw.textbbox((x1, y1 - 15), text)
        img_draw.rectangle(text_bbox, fill=color)
        img_draw.text((x1, y1 - 15), text, fill="white")
    return image

if __name__ == '__main__':
    
    # --- Load all models ---
    detector, config = load_object_detector()
    clip_model, clip_processor = load_clip_model()

    # --- Define your text prompts for zero-shot classification ---
    military_prompts = [
        "a tank", "a fighter jet", "a military helicopter", "a naval ship",
        "an armored personnel carrier", "a transport truck", "a soldier"
    ]
    
    # --- Run the pipeline on a test image ---
    test_image_path = "download (1).jpeg" # <--- CHANGE THIS
    
    try:
        result_image, boxes, labels, scores = run_zero_shot_detection(
            detector, clip_model, clip_processor, test_image_path, military_prompts, config
        )
        
        # --- Visualize and Save ---
        final_image = draw_results(result_image, boxes, labels, scores)
        final_image.save("zero_shot_output.jpg") # <-- THIS LINE IS FIXED
        print("✅ Zero-shot detection complete. Image saved to zero_shot_output.jpg")
        final_image.show()

    except FileNotFoundError:
        print(f"ERROR: Test image not found at '{test_image_path}'. Please update the path.")
    except Exception as e:
        print(f"An error occurred: {e}")