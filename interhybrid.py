import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import os
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import albumentations
from albumentations.pytorch import ToTensorV2

# CHANGED: Import the HybridCNN model
from models.cnn import HybridCNN

# --- Device Configuration ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- Load Face Detector ---
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# CHANGED: Initialize and load the HybridCNN model
# --- Load HybridCNN Model ---
print("Loading HybridCNN model...")
model = HybridCNN(num_qubits=4, num_classes=1).to(DEVICE)
# Ensure the path points to your trained hybrid model
checkpoint_path = './models/hybrid_cnn.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the trained hybrid model exists.")
    
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")


def predict(input_image:Image.Image, true_label:str):
    """Predict the label of the input_image and generate a GradCAM visualization."""
    
    # 1. Detect face in the image
    face = mtcnn(input_image)
    if face is None:
        return {'real': 0.0, 'fake': 0.0}, "No face detected!", None 
    
    # 2. Preprocess the face for the model
    face = face.unsqueeze(0) 
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    
    # For display purposes, create a normalized numpy version of the face
    face_np_display = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
    face_np_display = (face_np_display - face_np_display.min()) / (face_np_display.max() - face_np_display.min())

    # CHANGED: Apply the exact same normalization used during training the hybrid model
    # Convert tensor to numpy for albumentations
    face_np_for_transform = face.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    normalized_input_tensor = transform(image=face_np_for_transform)['image'].unsqueeze(0).to(DEVICE)

    # 3. Get model prediction
    with torch.no_grad():
        output = torch.sigmoid(model(normalized_input_tensor).squeeze(0))
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }

    # 4. Generate Grad-CAM visualization
    # CHANGED: Target a valid layer in HybridCNN, e.g., the second convolutional layer
    target_layers = [model.conv2]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)] 

    grayscale_cam = cam(input_tensor=normalized_input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(face_np_display, grayscale_cam, use_rgb=True)
    
    # Combine original face and CAM overlay for a blended image
    face_with_cam_overlay = Image.fromarray(visualization)

    return confidences, true_label, face_with_cam_overlay

# --- Create and Launch Gradio Interface ---
interface = gr.Interface(
    fn=predict,
    title="Hybrid Quantum-Classical Deepfake Detector",
    description="Upload an image to detect if the face is real or a deepfake. The interface also shows a Grad-CAM heatmap indicating which parts of the face the model focused on.",
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="True Label (for testing, optional)", value="unknown")
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="True Label Provided"),
        gr.Image(label="Detected Face with CAM Overlay", type="pil") 
    ],
    examples=[["examples/real_00001.jpg", "real"], ["examples/fake_00002.jpg", "fake"]]
)

interface.launch()