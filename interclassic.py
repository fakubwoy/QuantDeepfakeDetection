import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
# NEW IMPORTS FOR CORRECT NORMALIZATION
import albumentations
from albumentations.pytorch import ToTensorV2


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

use_checkpoint = True

if use_checkpoint:
    checkpoint = torch.load('./models/best_resnetinceptionv1.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load('./models/best_resnetinceptionv1.pth'))
model.to(DEVICE)
model.eval()


def predict(input_image:Image.Image, true_label:str):
    """Predict the label of the input_image and generate a GradCAM visualization."""
    face = mtcnn(input_image)
    if face is None:
        return {'real': 0.0, 'fake': 0.0}, "No face detected!", None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(224, 224), mode='bilinear', align_corners=False)

    # --- START: UNIFIED NORMALIZATION FIX ---
    # Convert tensor to numpy for albumentations
    face_np_for_transform = face.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    face_image_for_display = Image.fromarray(face_np_for_transform) # For Grad-CAM display later

    # Define the same transform used in training
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    # Apply the transform for model input
    input_tensor = transform(image=face_np_for_transform)['image'].unsqueeze(0).to(DEVICE)
    # --- END: UNIFIED NORMALIZATION FIX ---

    with torch.no_grad():
        output = torch.sigmoid(model(input_tensor).squeeze(0))
        real_prediction = 1 - output.item()
        fake_prediction = output.item()
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }

    # --- Grad-CAM Visualization (using the model input tensor) ---
    target_layers = [model.block8]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    face_np = np.array(face_image_for_display)
    face_np_normalized = face_np / 255.0 # Normalize to [0, 1] for visualization only
    face_np_normalized_uint8 = (face_np_normalized * 255).astype(np.uint8)

    visualization = show_cam_on_image(face_np_normalized, grayscale_cam, use_rgb=True)
    face_with_cam_overlay = Image.fromarray(cv2.addWeighted(face_np_normalized_uint8, 1, visualization, 0.5, 0))

    return confidences, true_label, face_with_cam_overlay


interface = gr.Interface(
    fn=predict,
    title="Classical Deepfake Detector (InceptionResnetV1)",
    description="Upload an image to detect if the face is real or a deepfake. The interface also shows a Grad-CAM heatmap indicating which parts of the face the model focused on for its prediction.",
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

interface.launch()