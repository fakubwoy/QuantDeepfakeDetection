import os
import torch
import wandb
import pandas as pd
import numpy as np
import albumentations
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Import the enhanced quantum model
from models.quantum_cnn import (
    ImprovedQuantumClassifier, 
    EnhancedQuantumLoss, 
    perform_enhanced_quantum_feature_selection
)
from dataset.faces import FaceDataset

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "./data"
TRAIN_BATCH_SIZE = 12  # Slightly smaller for better gradients
VAL_BATCH_SIZE = 16
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_WORKERS = 4
CHECKPOINT_PATH = "./models"

def get_datasets(data_folder, val_size=0.2):
    """Read and split the dataset with stratification."""
    faces_csv = os.path.join(data_folder, "faces", "faces.csv")
    faces_df = pd.read_csv(faces_csv)
    print(f"Original dataset length: {len(faces_df)}")
    
    # Check class distribution
    class_counts = faces_df['label'].value_counts()
    print(f"Class distribution: {dict(class_counts)}")
    
    train_df, val_df = train_test_split(
        faces_df, test_size=val_size, random_state=42, stratify=faces_df["label"]
    )
    print(f"Training dataset length ({1-val_size:.1%}): {len(train_df)}")
    print(f"Validation dataset length ({val_size:.1%}): {len(val_df)}")
    
    return train_df, val_df

def get_enhanced_transforms(img_width, img_height):
    """Enhanced transformations with better augmentation."""
    train_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.OneOf([
            albumentations.HorizontalFlip(p=1.0),
            albumentations.VerticalFlip(p=0.1),
            albumentations.RandomRotate90(p=0.1),
        ], p=0.5),
        albumentations.OneOf([
            albumentations.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-20, 20),
                p=1.0
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=1.0
            ),
        ], p=0.6),
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
        ], p=0.4),
        albumentations.OneOf([
            albumentations.GaussianBlur(blur_limit=(1, 3), p=1.0),
            albumentations.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        albumentations.GaussNoise(var_limit=(10, 50), p=0.2),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    
    val_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def extract_features_for_selection(model, dataloader, device, max_samples=2000):
    """
    Extract features with balanced sampling for better feature selection.
    """
    model.eval()
    all_features = []
    all_labels = []
    samples_per_class = {0: 0, 1: 0}  # Track samples per class
    max_per_class = max_samples // 2
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if all(samples_per_class[k] >= max_per_class for k in samples_per_class):
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            # Filter to maintain class balance
            keep_indices = []
            for i, label in enumerate(labels):
                label_val = int(label.item())
                if samples_per_class[label_val] < max_per_class:
                    keep_indices.append(i)
                    samples_per_class[label_val] += 1
            
            if not keep_indices:
                continue
            
            # Extract features for selected samples
            selected_images = images[keep_indices]
            selected_labels = labels[keep_indices]
            
            features = model.backbone(selected_images)
            
            all_features.append(features.cpu())
            all_labels.append(selected_labels.cpu())
    
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"Feature extraction completed: {features_tensor.shape[0]} samples")
    print(f"Class distribution in features: {torch.bincount(labels_tensor.long())}")
    
    return features_tensor, labels_tensor

def make_enhanced_model(config, selected_features=None):
    """Create the enhanced quantum model with improved optimizer."""
    model = ImprovedQuantumClassifier(
        num_classes=config.classes,
        selected_features=selected_features,
        freeze_backbone=config.freeze_backbone
    ).to(DEVICE)
    
    # Enhanced loss function
    criterion = EnhancedQuantumLoss(
        uncertainty_weight=config.uncertainty_weight,
        confidence_weight=config.confidence_weight
    )
    
    # Improved optimizer setup
    backbone_params = []
    attention_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'quantum_attention' in name:
            attention_params.append(param)
        else:
            classifier_params.append(param)
    
    # Different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config.learning_rate * 0.05, 'weight_decay': 1e-4},
        {'params': attention_params, 'lr': config.learning_rate * 0.5, 'weight_decay': 1e-5},
        {'params': classifier_params, 'lr': config.learning_rate, 'weight_decay': 1e-4}
    ])
    
    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )
    
    return model, criterion, optimizer, scheduler

def calculate_enhanced_metrics(outputs, labels, confidences=None):
    """Calculate comprehensive metrics including confidence calibration."""
    with torch.no_grad():
        probabilities = torch.sigmoid(outputs)
        predictions = probabilities > 0.5
        labels_bool = labels > 0.5
        
        # Basic metrics
        correct = (predictions == labels_bool).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        
        # Calculate precision, recall, F1
        tp = ((predictions == True) & (labels_bool == True)).sum().item()
        fp = ((predictions == True) & (labels_bool == False)).sum().item()
        fn = ((predictions == False) & (labels_bool == True)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(labels_bool.cpu().numpy(), probabilities.cpu().numpy())
        except:
            auc_roc = 0.5
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc
        }
        
        # Confidence calibration metrics
        if confidences is not None:
            correct_predictions = (predictions == labels_bool).float()
            confidence_scores = confidences.squeeze()
            
            # Expected Calibration Error (ECE)
            ece = calculate_ece(confidence_scores.cpu().numpy(), correct_predictions.cpu().numpy())
            metrics['ece'] = ece
        
        return metrics

def calculate_ece(confidences, accuracies, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def train_epoch_enhanced(model, dataloader, criterion, optimizer, device):
    """Enhanced training with confidence estimation."""
    model.train()
    running_loss = 0.0
    all_metrics = []
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        
        optimizer.zero_grad()
        
        # Forward pass with confidence
        outputs, confidences = model(images, return_confidence=True)
        
        # Get features for loss computation
        with torch.no_grad():
            features = model.backbone(images)
            if model.selected_features is not None:
                features = features[:, model.selected_features]
        
        # Enhanced loss calculation
        loss = criterion(outputs, labels, features, confidences)
        
        # Backward pass with gradient scaling
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Calculate metrics
        metrics = calculate_enhanced_metrics(outputs, labels, confidences)
        all_metrics.append(metrics)
        running_loss += loss.item()
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, avg_metrics

def validate_epoch_enhanced(model, dataloader, criterion, device):
    """Enhanced validation with confidence estimation."""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs, confidences = model(images, return_confidence=True)
            
            # Get features for loss computation
            features = model.backbone(images)
            if model.selected_features is not None:
                features = features[:, model.selected_features]
            
            loss = criterion(outputs, labels, features, confidences)
            
            metrics = calculate_enhanced_metrics(outputs, labels, confidences)
            all_metrics.append(metrics)
            running_loss += loss.item()
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, avg_metrics

def train_enhanced(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """Enhanced training loop with better monitoring and early stopping."""
    best_val_f1 = 0.0  # Use F1 score instead of loss for better balance
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    # Track training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'val_auc': [], 'val_ece': []
    }
    
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Train
        train_loss, train_metrics = train_epoch_enhanced(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_metrics = validate_epoch_enhanced(model, val_loader, criterion, DEVICE)
        
        # Update scheduler
        scheduler.step()
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_auc'].append(val_metrics['auc_roc'])
        if 'ece' in val_metrics:
            history['val_ece'].append(val_metrics['ece'])
        
        # Print comprehensive metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']*100:.2f}%, "
              f"F1: {train_metrics['f1_score']:.4f}, P: {train_metrics['precision']:.4f}, "
              f"R: {train_metrics['recall']:.4f}")
        
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%, "
              f"F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
        
        if 'ece' in val_metrics:
            print(f"Calibration - ECE: {val_metrics['ece']:.4f}")
        
        # Log to wandb with comprehensive metrics
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics['accuracy'] * 100,
            "val_accuracy": val_metrics['accuracy'] * 100,
            "train_f1": train_metrics['f1_score'],
            "val_f1": val_metrics['f1_score'],
            "train_precision": train_metrics['precision'],
            "val_precision": val_metrics['precision'],
            "train_recall": train_metrics['recall'],
            "val_recall": val_metrics['recall'],
            "val_auc_roc": val_metrics['auc_roc'],
            "learning_rate": current_lr
        }
        
        if 'ece' in val_metrics:
            log_dict["val_ece"] = val_metrics['ece']
        
        wandb.log(log_dict)
        
        # Enhanced model saving strategy
        improved = False
        
        # Primary criterion: F1 score (better balance between precision and recall)
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            improved = True
            save_suffix = "best_f1"
        
        # Secondary criterion: Loss (if F1 is close)
        elif abs(val_metrics['f1_score'] - best_val_f1) < 0.01 and val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
            save_suffix = "best_loss"
        
        if improved:
            patience_counter = 0
            
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            
            save_path = os.path.join(CHECKPOINT_PATH, f"enhanced_quantum_model_{save_suffix}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'train_acc': train_metrics['accuracy'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1_score'],
                'val_auc': val_metrics['auc_roc'],
                'selected_features': model.selected_features,
                'config': dict(config),
                'history': history
            }, save_path)
            
            print(f"✅ New best model saved: {save_suffix} "
                  f"(F1: {val_metrics['f1_score']:.4f}, AUC: {val_metrics['auc_roc']:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping with extended patience for quantum models
        if patience_counter >= max_patience:
            print(f"Early stopping after {max_patience} epochs without improvement")
            break
        
        # Learning rate monitoring
        if current_lr < 1e-7:
            print("Learning rate too small, stopping training")
            break
    
    # Save final model
    final_save_path = os.path.join(CHECKPOINT_PATH, "enhanced_quantum_model_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': val_loss,
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['f1_score'],
        'val_auc': val_metrics['auc_roc'],
        'selected_features': model.selected_features,
        'config': dict(config),
        'history': history
    }, final_save_path)
    
    return history

def model_pipeline_enhanced(hyperparameters):
    """Enhanced training pipeline with better feature selection and monitoring."""
    with wandb.init(project="enhanced-quantum-deepfake-detection", config=hyperparameters):
        config = wandb.config
        
        # Load datasets with enhanced transforms
        train_df, val_df = get_datasets(DATA_FOLDER)
        train_transform, val_transform = get_enhanced_transforms(IMG_WIDTH, IMG_HEIGHT)
        
        train_dataset = FaceDataset(DATA_FOLDER, train_df, train_transform)
        val_dataset = FaceDataset(DATA_FOLDER, val_df, val_transform)
        
        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        # Step 1: Create initial model for feature extraction
        print("🔧 Creating initial model for feature extraction...")
        initial_model, _, _, _ = make_enhanced_model(config)
        
        # Step 2: Enhanced quantum feature selection
        if config.use_quantum_feature_selection:
            print("🔮 Performing enhanced quantum feature selection...")
            features, labels = extract_features_for_selection(
                initial_model, train_loader, DEVICE, max_samples=2000
            )
            
            selected_features = perform_enhanced_quantum_feature_selection(
                features, labels, num_select=config.num_selected_features
            )
            
            # Log feature selection results
            wandb.log({
                "total_features": features.shape[1],
                "selected_features": len(selected_features),
                "feature_reduction_ratio": len(selected_features) / features.shape[1]
            })
        else:
            selected_features = None
            print("⏭️ Skipping quantum feature selection")
        
        # Step 3: Create enhanced model with selected features
        print("🚀 Creating enhanced quantum model...")
        model, criterion, optimizer, scheduler = make_enhanced_model(config, selected_features)
        model.selected_features = selected_features
        
        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Backbone frozen: {config.freeze_backbone}")
        
        if selected_features:
            print(f"   Selected features: {len(selected_features)}/512")
            print(f"   Feature reduction: {(1 - len(selected_features)/512)*100:.1f}%")
        
        # Log model info to wandb
        wandb.log({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
            "backbone_frozen": config.freeze_backbone
        })
        
        # Step 4: Enhanced training
        print("🎯 Beginning enhanced quantum training...")
        history = train_enhanced(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
        
        # Final evaluation and logging
        print("\n📈 Training Summary:")
        print(f"   Best Validation F1: {max(history['val_f1']):.4f}")
        print(f"   Best Validation AUC: {max(history['val_auc']):.4f}")
        print(f"   Final Validation Accuracy: {history['val_acc'][-1]*100:.2f}%")
        
        if history['val_ece']:
            print(f"   Final ECE (lower is better): {history['val_ece'][-1]:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Enhanced hyperparameters for better calibration and accuracy
    hyperparameters = dict(
        epochs=30,
        classes=1,
        learning_rate=0.0005,  # Slightly lower for better stability
        freeze_backbone=True,
        use_quantum_feature_selection=True,
        num_selected_features=320,  # Increased for better representation
        uncertainty_weight=0.003,   # Reduced for less regularization
        confidence_weight=0.015,    # Increased for better calibration
        dataset="Enhanced Faces Dataset",
        architecture="Enhanced-QuantumInceptionResnetV1",
        
        # Additional hyperparameters
        augmentation_strength="enhanced",
        optimizer="AdamW",
        scheduler="CosineAnnealingWarmRestarts"
    )
    
    print(f"🖥️ Running on device: {DEVICE.upper()}")
    print(f"🔬 Enhanced Quantum Training Configuration:")
    for key, value in hyperparameters.items():
        print(f"   {key}: {value}")
    
    model, history = model_pipeline_enhanced(hyperparameters)
    print("🎉 Enhanced quantum training completed!")