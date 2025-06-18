import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # Fixed typo from "backebds"
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Label map
label_map = {'stone': 1, 'rock': 2, 'mineral': 3, 'gem': 4, 'crystal': 5, 'mineral ore': 6}
class_names = ['__background__'] + list(label_map.keys())  # 0 is background

# Load model
model = fasterrcnn_mobilenet_v3_large_320_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(class_names))
model.load_state_dict(torch.load(os.getenv("MODEL_PTH"), map_location=device))
model.to(device)
model.eval()

# Define transforms
transform = A.Compose([
    A.Resize(height=512, width=512),
    ToTensorV2()
])

# Streamlit UI
st.title('🪨 Mineral Classifier App')
st.write('Upload an image to detect and classify minerals!')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        transformed = transform(image=image_np)
        input_tensor = transformed['image'].to(device).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            outputs = model(input_tensor)
            pred = outputs[0]  # First image in batch

        if len(pred['boxes']) == 0:
            st.warning("No objects detected.")
        else:
            # Get class with highest confidence
            top_idx = pred['scores'].argmax().item()
            pred_label = pred['labels'][top_idx].item()
            class_name = class_names[pred_label]
            st.success(f"Prediction: **{class_name}** (Confidence: {pred['scores'][top_idx]:.2f})")

