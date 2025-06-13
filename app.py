import os
import streamlit as st
import torch
import cv2
import numpy as np
from lmlt import LMLT

# Preprocessing function
def preprocess_image(image_path, lr_size=256):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    lr_img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
    lr_img = lr_img.astype(np.float32) / 255.0
    lr_img_tensor = torch.from_numpy(lr_img[np.newaxis, np.newaxis, :, :])
    return lr_img_tensor, img

# Function to perform inference
def perform_inference(model, lr_image_tensor):
    model.eval()
    with torch.no_grad():
        sr_image_tensor = model(lr_image_tensor)
    sr_image_np = sr_image_tensor.squeeze().cpu().numpy()
    sr_image_np = (sr_image_np * 255.0).astype(np.uint8)
    return sr_image_np

# Streamlit App
st.title("Image Super-Resolution using Vision Transformer (SRViT)")

st.write("Upload a Brain MRI image to convert it from Low Resolution (256x256) to High Resolution (512x512).")

# Load model (placeholder - you'll need to provide the actual trained model file)
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LMLT(in_channels=1, dim=36, n_blocks=8, upscaling_factor=2).to(device)
    # Assuming 'best_lmlt.pth' is available in the same directory as the Streamlit app
    # You would need to provide this file.
    try:
        model.load_state_dict(torch.load("models/best_lmlt.pth", map_location=torch.device('cpu')))
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model weights 'best_lmlt.pth' not found. Please ensure the model file is in the same directory.")
        st.stop()
    return model, device

model, device = load_model()

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the uploaded image
    try:
        lr_image_tensor, original_image_cv2 = preprocess_image("temp_image.jpg", lr_size=256)
        
        # Perform inference
        hr_image_np = perform_inference(model, lr_image_tensor.to(device))

        col1, col2 = st.columns(2)

        with col1:
            # Display the low resolution image
            st.subheader("Low Resolution Image (256x256)")
            st.image(
                cv2.resize(original_image_cv2, (256,256), interpolation=cv2.INTER_CUBIC),
                channels="GRAY",
                caption="Resized to 256x256",
                use_container_width=True
            )

        with col2:
        # Display both images side by side
            st.subheader("High Resolution Image (512x512)")
            st.image(
                hr_image_np,
                channels="GRAY",
                caption="Converted to 512x512",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
    finally:
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")