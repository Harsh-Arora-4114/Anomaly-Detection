import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import AutoEncoder

st.set_page_config(page_title="MVTec Anomaly Detection", layout="wide")

st.title("MVTec AD Anomaly Detection System")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Detect available categories
# -------------------------------
dataset_path = "dataset"

if not os.path.exists(dataset_path):
    st.error("Dataset folder not found!")
    st.stop()

categories = [folder for folder in os.listdir(dataset_path)
              if os.path.isdir(os.path.join(dataset_path, folder))]

if len(categories) == 0:
    st.error("No categories found inside dataset folder!")
    st.stop()

category = st.sidebar.selectbox("Select Category", categories)

model_file = f"{category}_model.pth"

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(model_path):
    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

if not os.path.exists(model_file):
    st.warning(f"Model file '{model_file}' not found. Train first.")
    st.stop()

model = load_model(model_file)

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

threshold = st.sidebar.slider("Anomaly Threshold", 0.001, 0.1, 0.02)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    img_resized = image.resize((256, 256))
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    error = torch.mean((output - img_tensor) ** 2).item()

    reconstructed = output.squeeze().cpu().numpy().transpose(1, 2, 0)

    with col2:
        st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)

    st.write(f"Reconstruction Error: {error:.6f}")

    if error > threshold:
        st.error("Anomaly Detected!")
    else:
        st.success("Normal Image")