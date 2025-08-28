import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import albumentations
from albumentations.pytorch import ToTensorV2

# Import HybridCNN
from models.cnn import HybridCNN

# --- Device ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- Face Detector ---
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

# --- Load HybridCNN Model ---
print("Loading HybridCNN...")
hybrid_model = HybridCNN(num_qubits=4, num_classes=1).to(DEVICE)
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
transform_hybrid = albumentations.Compose([
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# --- Hybrid Predictor ---
def predict_hybrid(input_image: Image.Image, true_label: str):
    face = mtcnn(input_image)
    if face is None:
        return {"real": 0.0, "fake": 0.0}, "⚠️ No face detected!", None, None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    face_np_display = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
    face_np_display = (face_np_display - face_np_display.min()) / (face_np_display.max() - face_np_display.min())
    original_face_img = Image.fromarray((face_np_display * 255).astype(np.uint8))

    face_np_for_transform = face.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    normalized_input_tensor = transform_hybrid(image=face_np_for_transform)['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = torch.sigmoid(hybrid_model(normalized_input_tensor).squeeze(0))
        real_pred = 1 - output.item()
        fake_pred = output.item()
        confidences = {"real": real_pred, "fake": fake_pred}

    target_layers = [hybrid_model.conv2]
    cam = GradCAM(model=hybrid_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=normalized_input_tensor, targets=[ClassifierOutputTarget(0)], eigen_smooth=True)[0, :]
    overlay_img = Image.fromarray(show_cam_on_image(face_np_display, grayscale_cam, use_rgb=True))

    return confidences, true_label, original_face_img, overlay_img

# --- Classic Predictor ---
def predict_classic(input_image: Image.Image, true_label: str):
    face = mtcnn(input_image)
    if face is None:
        return {"real": 0.0, "fake": 0.0}, "⚠️ No face detected!", None, None

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

    target_layers = [classic_model.block8]
    cam = GradCAM(model=classic_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=face, targets=[ClassifierOutputTarget(0)], eigen_smooth=True)[0, :]
    overlay_img = Image.fromarray(show_cam_on_image(face_np_display, grayscale_cam, use_rgb=True))

    return confidences, true_label, original_face_img, overlay_img

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Deepfake Detection: Hybrid vs Classic")
    gr.Markdown("Upload a face image to test with **Hybrid Quantum-Classical CNN** or **Classic InceptionResnetV1**.")

    with gr.Tab("Hybrid Model"):
        with gr.Row():
            with gr.Column(scale=1):
                hybrid_input = gr.Image(label="Upload Face Image", type="pil")
                hybrid_true_label = gr.Textbox(label="True Label (optional)", value="unknown")
                hybrid_btn = gr.Button("🔍 Analyze with Hybrid")
            with gr.Column(scale=1):
                hybrid_pred = gr.Label(label="Confidence Scores")
                hybrid_label = gr.Textbox(label="True Label Provided", interactive=False)
        with gr.Row():
            hybrid_face = gr.Image(label="Detected Face", type="pil")
            hybrid_overlay = gr.Image(label="Grad-CAM Overlay", type="pil")

        hybrid_btn.click(
            fn=predict_hybrid,
            inputs=[hybrid_input, hybrid_true_label],
            outputs=[hybrid_pred, hybrid_label, hybrid_face, hybrid_overlay]
        )

    with gr.Tab("Classic Model"):
        with gr.Row():
            with gr.Column(scale=1):
                classic_input = gr.Image(label="Upload Face Image", type="pil")
                classic_true_label = gr.Textbox(label="True Label (optional)", value="unknown")
                classic_btn = gr.Button("🔍 Analyze with Classic")
            with gr.Column(scale=1):
                classic_pred = gr.Label(label="Confidence Scores")
                classic_label = gr.Textbox(label="True Label Provided", interactive=False)
        with gr.Row():
            classic_face = gr.Image(label="Detected Face", type="pil")
            classic_overlay = gr.Image(label="Grad-CAM Overlay", type="pil")

        classic_btn.click(
            fn=predict_classic,
            inputs=[classic_input, classic_true_label],
            outputs=[classic_pred, classic_label, classic_face, classic_overlay]
        )

demo.launch()
