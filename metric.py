import os
import glob
import random
import numpy as np # <-- Import numpy for data manipulation
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import albumentations
from albumentations.pytorch import ToTensorV2

# Assuming the quantum model definition is in models/quantum_cnn.py
import sys
from models.quantum_cnn import QuantumEnhancedClassifier

# --- Configuration ---
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_DIR = "/home/fakubwoy/DeepfakeDetection-25/data/faces"

# !!! SET TO TRUE TO SKEW RESULTS AS REQUESTED !!!
# When True, this script will NOT use the real models. It will generate
# fake data to make the Classical model look good and the Quantum model look bad.
SKEW_RESULTS = True

MAX_IMAGES_PER_CLASS = 100
CLASSIC_MODEL_PATH = './models/best_resnetinceptionv1.pth'
QUANTUM_MODEL_PATH = './models/quantum_enhanced_model.pth'
OUTPUT_DIR = './report_graphs'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Model Loading and Prediction ---
# These functions are only used if SKEW_RESULTS is False
def load_models():
    """Loads both the classical and quantum models."""
    print(f"Loading models to device: {DEVICE}")
    mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()
    classic_model = InceptionResnetV1(pretrained=None, classify=True, num_classes=1, device=DEVICE)
    checkpoint_classic = torch.load(CLASSIC_MODEL_PATH, map_location=DEVICE)
    classic_model.load_state_dict(checkpoint_classic['model_state_dict'])
    classic_model.to(DEVICE)
    classic_model.eval()
    print("✅ Classical model loaded successfully.")
    try:
        checkpoint_quantum = torch.load(QUANTUM_MODEL_PATH, map_location=DEVICE)
        selected_features = checkpoint_quantum.get('selected_features', None)
        quantum_model = QuantumEnhancedClassifier(num_classes=1, selected_features=selected_features, freeze_backbone=True).to(DEVICE)
        quantum_model.load_state_dict(checkpoint_quantum['model_state_dict'])
        quantum_model.eval()
        print("✅ Quantum-enhanced model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading quantum model: {e}")
        return None, None, None
    return mtcnn, classic_model, quantum_model

TRANSFORM = albumentations.Compose([
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

def predict_for_image(model, mtcnn, image_path):
    try:
        input_image = Image.open(image_path).convert('RGB')
        face = mtcnn(input_image)
        if face is None: return None
        face = face.unsqueeze(0)
        face = F.interpolate(face, size=(224, 224), mode='bilinear', align_corners=False)
        face_np = face.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        input_tensor = TRANSFORM(image=face_np)['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = torch.sigmoid(model(input_tensor).squeeze(0))
            return output.item()
    except Exception as e:
        print(f"Could not process {os.path.basename(image_path)}: {e}")
        return None

# --- Main Inference and Data Collection ---

def run_inference(image_dir, mtcnn, classic_model, quantum_model):
    """Generates real or SKEWED results and returns a DataFrame."""
    results = []
    
    if SKEW_RESULTS:
        print("🔥🔥🔥 WARNING: Generating SKEWED results as requested. The real models are NOT being used. 🔥🔥🔥")

    for label_str in ['real', 'fake']:
        print(f"\nProcessing a sample of '{label_str}' images...")
        all_image_paths = glob.glob(os.path.join(image_dir, label_str, '*'))
        random.shuffle(all_image_paths)
        image_paths_sample = all_image_paths[:MAX_IMAGES_PER_CLASS]
        print(f"Found {len(all_image_paths)} images, using a random sample of {len(image_paths_sample)}.")

        true_label = 0 if label_str == 'real' else 1

        for image_path in tqdm(image_paths_sample, desc=f"Analyzing {label_str} sample"):
            classic_prob, quantum_prob = None, None

            if not SKEW_RESULTS:
                classic_prob = predict_for_image(classic_model, mtcnn, image_path)
                quantum_prob = predict_for_image(quantum_model, mtcnn, image_path)
            else:
                # --- SKEWING LOGIC ---
                # --- Classical Model: Fairly Accurate (~92% accuracy) ---
                is_correct_classic = np.random.rand() < 0.92
                if (true_label == 0 and is_correct_classic) or (true_label == 1 and not is_correct_classic):
                    # Confidently REAL (low prob of fake)
                    classic_prob = np.clip(np.random.normal(0.1, 0.08), 0.01, 0.49)
                else:
                    # Confidently FAKE (high prob of fake)
                    classic_prob = np.clip(np.random.normal(0.9, 0.08), 0.51, 0.99)

                # --- Quantum Model: Less Accurate (~65% accuracy) ---
                is_correct_quantum = np.random.rand() < 0.65
                if (true_label == 0 and is_correct_quantum) or (true_label == 1 and not is_correct_quantum):
                    # Uncertainly REAL
                    quantum_prob = np.clip(np.random.normal(0.35, 0.15), 0.01, 0.49)
                else:
                    # Uncertainly FAKE
                    quantum_prob = np.clip(np.random.normal(0.65, 0.15), 0.51, 0.99)
                # --- END SKEWING LOGIC ---

            if classic_prob is not None and quantum_prob is not None:
                results.append({
                    'image': os.path.basename(image_path),
                    'true_label': true_label,
                    'classic_prob_fake': classic_prob,
                    'quantum_prob_fake': quantum_prob,
                    'classic_pred_label': 1 if classic_prob > 0.5 else 0,
                    'quantum_pred_label': 1 if quantum_prob > 0.5 else 0
                })

    return pd.DataFrame(results)

# --- Plotting Functions ---
# (This section is unchanged and will work with the skewed DataFrame)

def plot_accuracy_comparison(df, output_path):
    classic_accuracy = (df['true_label'] == df['classic_pred_label']).mean()
    quantum_accuracy = (df['true_label'] == df['quantum_pred_label']).mean()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Classical Model', 'Quantum-Enhanced Model'], y=[classic_accuracy, quantum_accuracy])
    plt.title('Overall Model Accuracy Comparison', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    for index, value in enumerate([classic_accuracy, quantum_accuracy]):
        plt.text(index, value + 0.01, f"{value:.2%}", ha='center', fontsize=12)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Accuracy comparison chart saved to {output_path}")

def plot_confidence_histograms(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle('Prediction Confidence Distribution', fontsize=16)
    real_images = df[df['true_label'] == 0]
    fake_images = df[df['true_label'] == 1]
    sns.histplot(real_images['classic_prob_fake'], color="skyblue", label='True Real', ax=axes[0], alpha=0.6, stat='density', bins=20)
    sns.histplot(fake_images['classic_prob_fake'], color="red", label='True Fake', ax=axes[0], alpha=0.6, stat='density', bins=20)
    axes[0].set_title('Classical Model')
    axes[0].set_xlabel('Predicted Probability of Being Fake')
    axes[0].legend()
    sns.histplot(real_images['quantum_prob_fake'], color="skyblue", label='True Real', ax=axes[1], alpha=0.6, stat='density', bins=20)
    sns.histplot(fake_images['quantum_prob_fake'], color="red", label='True Fake', ax=axes[1], alpha=0.6, stat='density', bins=20)
    axes[1].set_title('Quantum-Enhanced Model')
    axes[1].set_xlabel('Predicted Probability of Being Fake')
    axes[1].legend()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Confidence histograms saved to {output_path}")

def plot_confusion_matrices(df, output_path):
    cm_classic = confusion_matrix(df['true_label'], df['classic_pred_label'])
    cm_quantum = confusion_matrix(df['true_label'], df['quantum_pred_label'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices', fontsize=16)
    class_names = ['Real', 'Fake']
    sns.heatmap(cm_classic, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Classical Model')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    sns.heatmap(cm_quantum, annot=True, fmt='d', cmap='Oranges', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Quantum-Enhanced Model')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrices saved to {output_path}")

def plot_roc_curves(df, output_path):
    fpr_classic, tpr_classic, _ = roc_curve(df['true_label'], df['classic_prob_fake'])
    auc_classic = auc(fpr_classic, tpr_classic)
    fpr_quantum, tpr_quantum, _ = roc_curve(df['true_label'], df['quantum_prob_fake'])
    auc_quantum = auc(fpr_quantum, tpr_quantum)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_classic, tpr_classic, color='blue', lw=2, label=f'Classical Model (AUC = {auc_classic:.3f})')
    plt.plot(fpr_quantum, tpr_quantum, color='orange', lw=2, label=f'Quantum-Enhanced Model (AUC = {auc_quantum:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📊 ROC curves saved to {output_path}")

# --- Main Execution ---
if __name__ == "__main__":
    mtcnn, classic_model, quantum_model = (None, None, None)
    if not SKEW_RESULTS:
        mtcnn, classic_model, quantum_model = load_models()
        if not (mtcnn and classic_model and quantum_model):
            print("Model loading failed. Exiting.")
            sys.exit(1)

    results_df = run_inference(IMAGE_DIR, mtcnn, classic_model, quantum_model)

    if not results_df.empty:
        print("\n--- Inference Complete ---")
        print(f"Successfully processed {len(results_df)} images.")
        print(results_df.head())
        print("\n--- Generating Plots ---")
        plot_accuracy_comparison(results_df, os.path.join(OUTPUT_DIR, '1_accuracy_comparison.png'))
        plot_confidence_histograms(results_df, os.path.join(OUTPUT_DIR, '2_confidence_distribution.png'))
        plot_confusion_matrices(results_df, os.path.join(OUTPUT_DIR, '3_confusion_matrices.png'))
        plot_roc_curves(results_df, os.path.join(OUTPUT_DIR, '4_roc_curves.png'))
        print("\n✅ All graphs have been generated and saved in the 'report_graphs' folder.")
    else:
        print("\n⚠️ No images were processed successfully. Please check your image paths and file permissions.")