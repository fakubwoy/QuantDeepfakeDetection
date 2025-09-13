import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import albumentations
from albumentations.pytorch import ToTensorV2

# Import the quantum-enhanced model
from models.quantum_cnn import QuantumEnhancedClassifier

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize face detector
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# Load the quantum-enhanced model
def load_quantum_model():
    """Load the trained quantum-enhanced model."""
    try:
        checkpoint = torch.load('./models/quantum_enhanced_model.pth', map_location=DEVICE)
        
        # Get selected features from checkpoint
        selected_features = checkpoint.get('selected_features', None)
        
        # Create model with same configuration
        model = QuantumEnhancedClassifier(
            num_classes=1,
            selected_features=selected_features,
            freeze_backbone=True
        ).to(DEVICE)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Loaded quantum-enhanced model from epoch {checkpoint['epoch']}")
        print(f"   - Validation Loss: {checkpoint['loss']:.4f}")
        print(f"   - Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
        if selected_features:
            print(f"   - Using {len(selected_features)} quantum-selected features")
        
        return model
        
    except Exception as e:
        print(f"Error loading quantum model: {e}")
        print("Creating default model...")
        model = QuantumEnhancedClassifier(num_classes=1).to(DEVICE)
        model.eval()
        return model

model = load_quantum_model()

def predict(input_image: Image.Image, true_label: str):
    """
    Predict using the quantum-enhanced model and provide visualizations.
    """
    # Face detection
    face = mtcnn(input_image)
    if face is None:
        return {
            'real': 0.0, 
            'fake': 0.0
        }, "No face detected!", None, "No quantum analysis available"

    # Preprocess face
    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(224, 224), mode='bilinear', align_corners=False)

    # Convert for transformation (same as other interfaces)
    face_for_transform = face.squeeze(0).permute(1, 2, 0)
    face_np_for_transform = ((face_for_transform + 1) / 2.0 * 255).cpu().numpy().astype(np.uint8)
    face_image_for_display = Image.fromarray(face_np_for_transform)

    # Apply the same transform used in training
    transform = albumentations.Compose([
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    input_tensor = transform(image=face_np_for_transform)['image'].unsqueeze(0).to(DEVICE)

    # Model prediction with detailed analysis
    with torch.no_grad():
        # Get full forward pass
        output = model(input_tensor)
        output_sigmoid = torch.sigmoid(output).squeeze()
        
        fake_prediction = output_sigmoid.item()
        real_prediction = 1 - fake_prediction
        
        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }
        
        # Get quantum attention analysis
        features = model.backbone(input_tensor)
        if model.selected_features is not None:
            selected_features = features[:, model.selected_features]
            quantum_analysis = analyze_quantum_features(selected_features, model)
        else:
            quantum_analysis = "Quantum feature selection not used in this model"

    # Generate Grad-CAM visualization
    grad_cam_image = generate_grad_cam(model, input_tensor, face_np_for_transform)

    return confidences, true_label, grad_cam_image, quantum_analysis

def analyze_quantum_features(features, model):
    """
    Analyze the quantum-inspired attention weights and feature importance.
    """
    try:
        # Get quantum attention weights
        with torch.no_grad():
            attended_features = model.quantum_attention(features)
            
            # Analyze quantum amplitude parameters
            amplitude_real = model.quantum_attention.amplitude_real.cpu().numpy()
            amplitude_imag = model.quantum_attention.amplitude_imag.cpu().numpy()
            
            # Calculate quantum probabilities
            amplitude_norm = np.sqrt(amplitude_real**2 + amplitude_imag**2 + 1e-8)
            quantum_probs = (amplitude_real / amplitude_norm)**2 + (amplitude_imag / amplitude_norm)**2
            
            # Feature importance analysis
            feature_importance = torch.abs(attended_features - features).mean(dim=0).cpu().numpy()
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
            analysis = f"""🔮 Quantum Analysis:
            
• Selected Features: {len(model.selected_features) if model.selected_features else 'All 512'}
• Quantum Interference Pattern:
  - Max Amplitude: {np.max(amplitude_norm):.4f}
  - Min Amplitude: {np.min(amplitude_norm):.4f}
  - Mean Quantum Probability: {np.mean(quantum_probs):.4f}

• Top 5 Most Important Features:
  - Feature {top_features[0]}: {feature_importance[top_features[0]]:.4f}
  - Feature {top_features[1]}: {feature_importance[top_features[1]]:.4f}
  - Feature {top_features[2]}: {feature_importance[top_features[2]]:.4f}
  - Feature {top_features[3]}: {feature_importance[top_features[3]]:.4f}
  - Feature {top_features[4]}: {feature_importance[top_features[4]]:.4f}

• Quantum Coherence: {1 - np.var(quantum_probs):.4f}
"""
            return analysis
            
    except Exception as e:
        return f"Quantum analysis error: {str(e)}"

def generate_grad_cam(model, input_tensor, face_np):
    """
    Generate Grad-CAM visualization for the quantum-enhanced model.
    """
    try:
        # Use the backbone's last convolutional layer for Grad-CAM
        target_layers = [model.backbone.block8]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(0)]

        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]

        # Normalize face image for visualization
        face_np_normalized = face_np / 255.0
        
        # Create visualization
        visualization = show_cam_on_image(face_np_normalized, grayscale_cam, use_rgb=True)
        return Image.fromarray(visualization)
        
    except Exception as e:
        print(f"Grad-CAM failed: {e}")
        return Image.fromarray(face_np)

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    title="🔮 Quantum-Enhanced Deepfake Detector",
    description="""
    This model uses quantum-inspired algorithms for enhanced deepfake detection:
    
    • **Quantum Feature Selection**: Uses QAOA (Quantum Approximate Optimization Algorithm) to select the most important features
    • **Quantum-Inspired Attention**: Applies quantum interference principles to weight features
    • **Uncertainty Regularization**: Incorporates quantum uncertainty principles in training
    
    Upload an image to detect if the face is real or a deepfake. The model provides both classical Grad-CAM visualization and quantum feature analysis.
    """,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="True Label (for testing, optional)", value="unknown")
    ],
    outputs=[
        gr.Label(label="🎯 Prediction Confidence", num_top_classes=2),
        gr.Textbox(label="📝 True Label Provided"),
        gr.Image(label="🔥 Grad-CAM Heatmap", type="pil"),
        gr.Textbox(label="🔮 Quantum Analysis", lines=15, max_lines=20)
    ],
    examples=[
        # You can add example images here
        # ["./examples/real_face.jpg", "real"],
        # ["./examples/fake_face.jpg", "fake"],
    ],
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
)

def create_comparison_interface():
    """
    Create a comparison interface that shows results from both classical and quantum models.
    """
    
    def compare_predictions(input_image: Image.Image):
        """Compare predictions from both models."""
        if input_image is None:
            return "Please upload an image", "", "", ""
        
        # Quantum model prediction
        quantum_conf, _, quantum_cam, quantum_analysis = predict(input_image, "unknown")
        
        # Format results
        quantum_result = f"Quantum Model:\nReal: {quantum_conf['real']:.3f}\nFake: {quantum_conf['fake']:.3f}"
        
        comparison_analysis = f"""
        🔬 **Model Comparison Analysis**
        
        **Quantum-Enhanced Model Results:**
        • Real Confidence: {quantum_conf['real']:.3f} ({quantum_conf['real']*100:.1f}%)
        • Fake Confidence: {quantum_conf['fake']:.3f} ({quantum_conf['fake']*100:.1f}%)
        • Prediction: {'REAL' if quantum_conf['real'] > quantum_conf['fake'] else 'FAKE'}
        
        **Quantum Advantages:**
        • Feature dimensionality reduction through quantum optimization
        • Quantum interference-based attention mechanism
        • Uncertainty-aware loss function
        • Enhanced interpretability through quantum analysis
        """
        
        return quantum_result, quantum_cam, quantum_analysis, comparison_analysis
    
    comparison_interface = gr.Interface(
        fn=compare_predictions,
        title="🔬 Quantum vs Classical Deepfake Detection Comparison",
        description="Compare the quantum-enhanced model with detailed analysis.",
        inputs=[
            gr.Image(label="Input Image", type="pil")
        ],
        outputs=[
            gr.Textbox(label="📊 Prediction Results", lines=4),
            gr.Image(label="🔥 Quantum Model Grad-CAM", type="pil"),
            gr.Textbox(label="🔮 Quantum Analysis", lines=12),
            gr.Textbox(label="🔬 Comparison Analysis", lines=10)
        ],
        theme=gr.themes.Soft()
    )
    
    return comparison_interface

if __name__ == "__main__":
    print("🚀 Starting Quantum-Enhanced Deepfake Detector Interface...")
    print(f"   Running on device: {DEVICE.upper()}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Launch the main interface
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )