import streamlit as st
from PIL import Image, ImageDraw
import torch
import yaml
from torchvision import transforms

from modules.models import LightweightMultiModalDETR

@st.cache_resource
def load_resources(config_path="config.yaml"):
    """Loads config and model, caching them for performance."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model = LightweightMultiModalDETR(config)
        model.load_state_dict(torch.load(config['demo']['model_path'], map_location=torch.device('cpu')))
        model.eval()
        
        class_map = {i: name for i, name in enumerate(config['data']['class_names'])}
        return model, config, class_map
    except FileNotFoundError:
        return None, None, None

def draw_boxes(image, boxes, labels, scores, class_map):
    """Draws styled bounding boxes on the image."""
    img_draw = ImageDraw.Draw(image)
    colors = ['#FF3B30', '#FF9500', '#FFCC00', '#34C759', '#007AFF', '#5856D6', '#AF52DE']
    
    for box, label, score in zip(boxes, labels, scores):
        w, h = image.size
        x_c, y_c, w_b, h_b = box * torch.tensor([w, h, w, h])
        x1, y1, x2, y2 = x_c - w_b/2, y_c - h_b/2, x_c + w_b/2, y_c + h_b/2
        
        color = colors[label % len(colors)]
        img_draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        class_name = class_map.get(label, f"Class {label}")
        text = f"{class_name} ({score:.2f})"
        
        text_bbox = img_draw.textbbox((x1, y1 - 15), text)
        img_draw.rectangle(text_bbox, fill=color)
        img_draw.text((x1, y1 - 15), text, fill="white")
        
    return image

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Multimodal Detection Demo")
st.title("🛰️ Lightweight Object Detection Demo")
st.write("This model was trained on a combined dataset of AU-AIR (RGB) and HIT-UAV (Thermal).")

model, config, CLASS_MAP = load_resources()

if model is None:
    st.error(f"Model file not found at `{config['demo']['model_path']}`. Please run `python main.py` to train and save the model first.")
else:
    st.sidebar.header("⚙️ Controls")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, config['demo']['confidence_threshold'], 0.05)
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Input Image")
        uploaded_file = st.file_uploader("Upload an RGB or Thermal image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Determine appropriate normalization
        is_grayscale = True if image.mode == 'L' or image.getextrema()[0] == image.getextrema()[1] else False
        if is_grayscale:
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        preprocess = transforms.Compose([transforms.Resize(config['data']['img_size']), transforms.ToTensor(), normalize])
        img_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Post-process the output
        probs = outputs['pred_logits'][0].softmax(-1)
        scores, labels = probs.max(-1)
        keep = scores > conf_thresh
        
        top_boxes = outputs['pred_boxes'][0][keep]
        top_labels = labels[keep].cpu().numpy()
        top_scores = scores[keep].cpu().numpy()
        
        with col2:
            st.header("Detection Results")
            if len(top_boxes) > 0:
                output_image = draw_boxes(image.copy(), top_boxes, top_labels, top_scores, CLASS_MAP)
                st.image(output_image, caption=f"Found {len(top_boxes)} objects.", use_column_width='auto')
            else:
                st.image(image, caption="No objects detected above the confidence threshold.", use_column_width=True)