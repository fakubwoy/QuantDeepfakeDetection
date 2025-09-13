import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import albumentations
from albumentations.pytorch import ToTensorV2

# Import your hybrid model
from models.cnn import ImprovedHybridQNN

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Face detection
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# Load hybrid model
model = ImprovedHybridQNN(
    num_qubits=4,
    reps=1,
    num_classes=1
).to(DEVICE)

# Load the trained weights
use_checkpoint = True

if use_checkpoint:
    checkpoint = torch.load('./models/hybrid_cnn.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
else:
    model.load_state_dict(torch.load('./models/hybrid_cnn.pth'))

model.eval()

def predict(input_image: Image.Image, true_label: str):
    """Predict the label of the input_image using the hybrid quantum-classical model."""
    
    # Face detection
    face = mtcnn(input_image)
    if face is None:
        return {'real': 0.0, 'fake': 0.0}, "No face detected!", None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(224, 224), mode='bilinear', align_corners=False)

    # --- START: CORRECTED PREPROCESSING ---
    # 1. Squeeze the batch dimension and move channel to the end for numpy conversion.
    face_for_transform = face.squeeze(0).permute(1, 2, 0)
    
    # 2. Correctly convert tensor range from [-1, 1] to [0, 255] for albumentations.
    face_np_for_transform = ((face_for_transform + 1) / 2.0 * 255).cpu().numpy().astype(np.uint8)
    
    # This image is now correctly formatted for display and transformation.
    face_image_for_display = Image.fromarray(face_np_for_transform)

    # 3. Define the EXACT same transform used in training.
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    # 4. Apply the transform for model input.
    input_tensor = transform(image=face_np_for_transform)['image'].unsqueeze(0).to(DEVICE)
    # --- END: CORRECTED PREPROCESSING ---

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        output_sigmoid = torch.sigmoid(output).squeeze()
        
        fake_prediction = output_sigmoid.item()
        real_prediction = 1 - fake_prediction
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }

    # --- Grad-CAM Visualization ---
    try:
        target_layers = [model.feature_extractor.block8] 
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]

        face_np_normalized = face_np_for_transform / 255.0

        visualization = show_cam_on_image(face_np_normalized, grayscale_cam, use_rgb=True)
        face_with_cam_overlay = Image.fromarray(visualization)
        
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        face_with_cam_overlay = face_image_for_display

    return confidences, true_label, face_with_cam_overlay


interface = gr.Interface(
    fn=predict,
    title="Hybrid Quantum-Classical Deepfake Detector",
    description="Upload an image to detect if the face is real or a deepfake using a hybrid quantum-classical neural network. The interface shows a Grad-CAM heatmap indicating which parts of the face the model focused on.",
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="True Label (for testing, optional)", value="unknown")
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="True Label Provided"),
        gr.Image(label="Detected Face with CAM", type="pil")
    ],
)

if __name__ == "__main__":
    interface.launch()