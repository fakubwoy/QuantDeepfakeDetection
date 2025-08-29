import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import time  # Added for processing time measurement
# CORRECTED: Import ScoreCAM instead of GradCAM for the hybrid model
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations
from albumentations.pytorch import ToTensorV2
import cv2 # Added for reliable image resizing

from models.cnn import HybridResnetQNN

# --- Device ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- Face Detector ---
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

# --- Load HybridResnetQNN Model ---
print("Loading HybridResnetQNN...")
hybrid_model = HybridResnetQNN(num_qubits=4, num_classes=1).to(DEVICE)
hybrid_ckpt = './models/hybrid_cnn.pth'

if os.path.exists(hybrid_ckpt) and os.path.getsize(hybrid_ckpt) > 1000:  # >1KB means likely valid
    try:
        checkpoint = torch.load(hybrid_ckpt, map_location=DEVICE)
        hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        print("Hybrid model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not load Hybrid model: {e}")
        print("👉 Using untrained Hybrid model for UI testing...")
else:
    print("⚠️ hybrid_cnn.pth is missing or invalid (too small).")
    print("👉 Using untrained Hybrid model for UI testing...")

# --- Load Classic InceptionResnetV1 Model ---
print("Loading Classic InceptionResnetV1...")
classic_model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=DEVICE)
classic_ckpt = './models/resnetinceptionv1_epoch_32.pth'

if os.path.exists(classic_ckpt) and os.path.getsize(classic_ckpt) > 1000:
    try:
        checkpoint = torch.load(classic_ckpt, map_location=DEVICE)
        classic_model.load_state_dict(checkpoint['model_state_dict'])
        print("Classic model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Could not load Classic model: {e}")
        print("👉 Using untrained Classic model for UI testing...")
else:
    print("⚠️ Classic checkpoint is missing or invalid.")
    print("👉 Using untrained Classic model for UI testing...")

classic_model.to(DEVICE).eval()

# --- Preprocessing for Hybrid ---
# This transform now expects a 224x224 input image
transform_hybrid = albumentations.Compose([
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- Hybrid Predictor (FULLY CORRECTED with processing time) ---
def predict_hybrid(input_image: Image.Image):
    start_time = time.time()  # Start timing
    
    # 1. Detect face
    face_tensor = mtcnn(input_image)
    if face_tensor is None:
        processing_time = time.time() - start_time
        return {"real": 0.0, "fake": 0.0}, f"⚠️ No face detected! (Processing time: {processing_time:.2f}s)", None, None

    # 2. Convert detected face tensor to a NumPy array for transformations
    face_denormalized = (face_tensor.permute(1, 2, 0).cpu() + 1) * 127.5
    face_np = face_denormalized.numpy().astype(np.uint8)

    # 3. Resize to the model's expected input size (224x224)
    face_np_resized_display = cv2.resize(face_np, (224, 224))
    detected_face_img = Image.fromarray(face_np_resized_display)

    # 4. Preprocess the resized face for the model
    normalized_input_tensor = transform_hybrid(image=face_np_resized_display)['image'].unsqueeze(0).to(DEVICE)

    # 5. Get model prediction
    with torch.no_grad():
        output = torch.sigmoid(hybrid_model(normalized_input_tensor).squeeze())
        real_pred = 1 - output.item()
        fake_pred = output.item()
        confidences = {"real": real_pred, "fake": fake_pred}

    # 6. Generate CAM Overlay using ScoreCAM (gradient-free)
    target_layers = [hybrid_model.feature_extractor.layer4[-1]]
    # Use ScoreCAM because GradCAM fails due to broken gradient flow in the hybrid model
    cam = ScoreCAM(model=hybrid_model, target_layers=target_layers)

    # The numpy array for the overlay function needs to be in [0, 1] float format
    face_np_float_display = face_np_resized_display / 255.0

    grayscale_cam = cam(input_tensor=normalized_input_tensor, targets=[ClassifierOutputTarget(0)], eigen_smooth=True)[0, :]
    overlay_img = Image.fromarray(show_cam_on_image(face_np_float_display, grayscale_cam, use_rgb=True))

    processing_time = time.time() - start_time  # Calculate processing time
    
    # Add processing time to the result message
    result_text = f"Analysis complete! (Processing time: {processing_time:.2f}s)"
    
    return confidences, result_text, detected_face_img, overlay_img


# --- Classic Predictor (Modified with processing time) ---
def predict_classic(input_image: Image.Image):
    start_time = time.time()  # Start timing
    
    face = mtcnn(input_image)
    if face is None:
        processing_time = time.time() - start_time
        return {"real": 0.0, "fake": 0.0}, f"⚠️ No face detected! (Processing time: {processing_time:.2f}s)", None, None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    face_np_display = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    face_np_display = (face_np_display - face_np_display.min()) / (face_np_display.max() - face_np_display.min())
    original_face_img = Image.fromarray((face_np_display * 255).astype(np.uint8))

    face = face.to(DEVICE).to(torch.float32) / 255.0

    with torch.no_grad():
        output = torch.sigmoid(classic_model(face).squeeze(0))
        real_pred = 1 - output.item()
        fake_pred = output.item()
        confidences = {"real": real_pred, "fake": fake_pred}

    # GradCAM is fine for the standard classic model
    target_layers = [classic_model.block8]
    cam = GradCAM(model=classic_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=face, targets=[ClassifierOutputTarget(0)], eigen_smooth=True)[0, :]
    overlay_img = Image.fromarray(show_cam_on_image(face_np_display, grayscale_cam, use_rgb=True))

    processing_time = time.time() - start_time  # Calculate processing time
    
    # Add processing time to the result message
    result_text = f"Analysis complete! (Processing time: {processing_time:.2f}s)"

    return confidences, result_text, original_face_img, overlay_img


# --- Modern Dark Theme CSS (FIXED) ---
modern_css = """
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables for consistent theming */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #111111;
    --bg-tertiary: #1a1a1a;
    --bg-card: #161616;
    --border-color: #2a2a2a;
    --border-hover: #3a3a3a;
    --accent-primary: #6366f1;
    --accent-secondary: #8b5cf6;
    --accent-success: #10b981;
    --accent-warning: #f59e0b;
    --accent-danger: #ef4444;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
}

/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    min-height: 100vh !important;
}

.app {
    background: var(--bg-primary) !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 2rem !important;
}

/* Header styling */
.gradio-container > div:first-child {
    text-align: center !important;
    margin-bottom: 3rem !important;
    padding: 3rem 2rem !important;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow-xl) !important;
}

/* Title styling - FIXED to be visible */
.gradio-container h1 {
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: var(--accent-primary) !important;
    margin-bottom: 1rem !important;
    letter-spacing: -0.02em !important;
    display: inline-block !important;
    text-shadow: 0 2px 10px rgba(99, 102, 241, 0.5);
}

.gradio-container h1 + p {
    font-size: 1.125rem !important;
    color: var(--text-secondary) !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    max-width: 600px !important;
    margin: 0 auto !important;
}

/* Tab navigation */
.tab-nav {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    margin-bottom: 2rem !important;
    display: flex !important;
    gap: 4px !important;
}

.tab-nav button {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    flex: 1 !important;
    position: relative !important;
}

.tab-nav button:hover {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
}

.tab-nav button.selected {
    background: var(--accent-primary) !important;
    color: white !important;
    box-shadow: var(--shadow-md) !important;
}

/* Card containers */
.gradio-row {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    margin-bottom: 1.5rem !important;
    box-shadow: var(--shadow-lg) !important;
}

/* Model badges */
.gradio-row:has(.gr-prose h3) {
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
    border: none !important;
    padding: 1rem 0 !important;
    margin-bottom: 2rem !important;
}

.gradio-row h3 {
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    text-align: center !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Input sections */
.gr-image, .gr-file-input {
    background: var(--bg-secondary) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    min-height: 200px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.gr-image:hover, .gr-file-input:hover {
    border-color: var(--accent-primary) !important;
    background: var(--bg-tertiary) !important;
}

.gr-image img {
    border-radius: 8px !important;
    box-shadow: var(--shadow-md) !important;
}

/* Upload text */
.gr-image .gr-prose, .gr-file-input .gr-prose {
    color: var(--text-secondary) !important;
    text-align: center !important;
}

/* Buttons */
.gr-button {
    background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow-md) !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-xl) !important;
    opacity: 0.9 !important;
}

.gr-button:active {
    transform: translateY(0) !important;
}

/* Labels */
.gr-label {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Results - Label component for confidence */
.gr-label-output {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.9rem !important;
}

/* Status text styling */
.gr-textbox {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
}

/* Output images */
.gr-image .gr-image-preview {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-lg) !important;
    border: 1px solid var(--border-color) !important;
}

/* Loading states */
.gr-loading {
    background: var(--bg-tertiary) !important;
    border-radius: 8px !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px !important;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary) !important;
}

::-webkit-scrollbar-thumb {
    background: var(--border-color) !important;
    border-radius: 4px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--border-hover) !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .app {
        padding: 1rem !important;
    }
    
    .gradio-container h1 {
        font-size: 2rem !important;
    }
    
    .gradio-row {
        padding: 1.5rem !important;
    }
    
    .tab-nav button {
        padding: 10px 16px !important;
        font-size: 0.8rem !important;
    }
}

/* Animation for smooth transitions */
* {
    transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease !important;
}

/* Focus states for accessibility */
button:focus, input:focus {
    outline: 2px solid var(--accent-primary) !important;
    outline-offset: 2px !important;
}
"""

# --- Modern Gradio UI (Updated with status display) ---
with gr.Blocks(css=modern_css, theme=gr.themes.Base()) as demo:
    # Header Section
    with gr.Row():
        with gr.Column():
            gr.Markdown("# 🔬 AI Deepfake Detector")
            gr.Markdown("Advanced neural network analysis using quantum-classical hybrid and traditional CNN architectures")
    
    # Main Interface
    with gr.Tabs():
        # Hybrid Model Tab
        with gr.Tab("🚀 Quantum-Hybrid CNN"):
            with gr.Row():
                gr.Markdown("### 🧪 Quantum-Classical Hybrid Model")
                
            with gr.Row():
                # Input section
                with gr.Column(scale=1):
                    hybrid_input = gr.Image(
                        label="📸 Upload Face Image", 
                        type="pil",
                        height=320
                    )
                    hybrid_btn = gr.Button("🔍 Analyze with Hybrid AI")
                
                # Results section
                with gr.Column(scale=1):
                    hybrid_pred = gr.Label(label="📊 Confidence Analysis")
                    hybrid_status = gr.Textbox(
                        label="⚡ Processing Status",
                        value="Ready for analysis",
                        interactive=False
                    )
            
            # Visualization section
            with gr.Row():
                with gr.Column():
                    hybrid_face = gr.Image(
                        label="👤 Detected Face", 
                        type="pil",
                        height=280
                    )
                with gr.Column():
                    hybrid_overlay = gr.Image(
                        label="🔥 Neural Attention Heatmap", 
                        type="pil",
                        height=280
                    )

            hybrid_btn.click(
                fn=predict_hybrid,
                inputs=[hybrid_input],
                outputs=[hybrid_pred, hybrid_status, hybrid_face, hybrid_overlay]
            )

        # Classic Model Tab  
        with gr.Tab("🏛️ Classical CNN"):
            with gr.Row():
                gr.Markdown("### ⚡ Traditional Inception-ResNet Model")
                
            with gr.Row():
                # Input section
                with gr.Column(scale=1):
                    classic_input = gr.Image(
                        label="📸 Upload Face Image", 
                        type="pil",
                        height=320
                    )
                    classic_btn = gr.Button("🔍 Analyze with Classic AI")
                
                # Results section
                with gr.Column(scale=1):
                    classic_pred = gr.Label(label="📊 Confidence Analysis")
                    classic_status = gr.Textbox(
                        label="⚡ Processing Status",
                        value="Ready for analysis",
                        interactive=False
                    )
            
            # Visualization section
            with gr.Row():
                with gr.Column():
                    classic_face = gr.Image(
                        label="👤 Detected Face", 
                        type="pil",
                        height=280
                    )
                with gr.Column():
                    classic_overlay = gr.Image(
                        label="🔥 Neural Attention Heatmap", 
                        type="pil",
                        height=280
                    )

            classic_btn.click(
                fn=predict_classic,
                inputs=[classic_input],
                outputs=[classic_pred, classic_status, classic_face, classic_overlay]
            )

demo.launch()
