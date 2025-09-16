import os
import torch
import wandb
import pandas as pd
import numpy as np
import albumentations
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import torch.nn.functional as F

from models.quantum_cnn import (
    BalancedQuantumClassifier, 
    AdaptiveFocalLoss, 
    perform_advanced_quantum_feature_selection,
    MixupAugmentation
)
from dataset.faces import FaceDataset

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "./data"
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 20
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_WORKERS = 6
CHECKPOINT_PATH = "./models"

def create_balanced_sampler(labels, oversample_minority=True):
    label_counts = Counter(labels)
    total_samples = len(labels)
    
    if oversample_minority:
        max_count = max(label_counts.values())
        weights = {label: max_count / count for label, count in label_counts.items()}
    else:
        min_count = min(label_counts.values())
        weights = {label: min_count / count for label, count in label_counts.items()}
    
    sample_weights = [weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples if not oversample_minority else max(label_counts.values()) * len(label_counts),
        replacement=True
    )

def get_stratified_datasets(data_folder, val_size=0.15, n_folds=1):
    faces_csv = os.path.join(data_folder, "faces", "faces.csv")
    faces_df = pd.read_csv(faces_csv)
    
    print(f"Dataset size: {len(faces_df)}")
    class_dist = faces_df['label'].value_counts().to_dict()
    print(f"Original distribution: {class_dist}")
    imbalance_ratio = max(class_dist.values()) / min(class_dist.values())
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if n_folds == 1:
        skf = StratifiedKFold(n_splits=int(1/val_size), shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(faces_df, faces_df['label']))
        train_df = faces_df.iloc[train_idx].reset_index(drop=True)
        val_df = faces_df.iloc[val_idx].reset_index(drop=True)
        return [(train_df, val_df)]
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = []
        for train_idx, val_idx in skf.split(faces_df, faces_df['label']):
            train_df = faces_df.iloc[train_idx].reset_index(drop=True)
            val_df = faces_df.iloc[val_idx].reset_index(drop=True)
            folds.append((train_df, val_df))
        return folds

def get_advanced_transforms(img_width, img_height, is_real_heavy=False):
    base_transforms = [
        albumentations.Resize(img_width, img_height),
        albumentations.RandomResizedCrop((img_height, img_width), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.5),
    ]
    
    real_augmentations = [
        albumentations.OneOf([
            albumentations.HorizontalFlip(p=1.0),
            albumentations.VerticalFlip(p=0.3),
            albumentations.RandomRotate90(p=0.3),
        ], p=0.7),
        
        albumentations.OneOf([
            albumentations.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=1.0
            ),
            albumentations.Affine(
                scale=(0.85, 1.15), translate_percent=(-0.15, 0.15), 
                rotate=(-25, 25), shear=(-10, 10), p=1.0
            ),
            albumentations.ElasticTransform(alpha=50, sigma=5, p=1.0),
        ], p=0.8 if is_real_heavy else 0.6),
        
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0
            ),
            albumentations.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1.0
            ),
            albumentations.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0
            ),
        ], p=0.6),
        
        albumentations.OneOf([
            albumentations.GaussianBlur(blur_limit=(1, 5), p=1.0),
            albumentations.MedianBlur(blur_limit=5, p=1.0),
            albumentations.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.4),
        
        albumentations.OneOf([
            albumentations.GaussNoise(var_limit=(15, 75), p=1.0),
            albumentations.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            albumentations.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1.0),
        ], p=0.3),
        
        albumentations.OneOf([
            albumentations.Cutout(num_holes=3, max_h_size=16, max_w_size=16, p=1.0),
            albumentations.CoarseDropout(
                max_holes=5, max_height=20, max_width=20, 
                min_holes=1, min_height=8, min_width=8, p=1.0
            ),
        ], p=0.2),
    ]
    
    if is_real_heavy:
        train_transform = albumentations.Compose(
            base_transforms + real_augmentations + [
                albumentations.OneOf([
                    albumentations.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    albumentations.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
                ], p=0.3),
                albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        )
    else:
        standard_augmentations = [
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomRotate90(p=0.2),
            ], p=0.5),
            
            albumentations.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4
            ),
            
            albumentations.OneOf([
                albumentations.GaussianBlur(blur_limit=(1, 3), p=1.0),
                albumentations.GaussNoise(var_limit=(10, 50), p=1.0),
            ], p=0.2),
        ]
        
        train_transform = albumentations.Compose(
            base_transforms + standard_augmentations + [
                albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        )
    
    val_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

class BalancedFaceDataset(FaceDataset):
    def __init__(self, data_folder, dataframe, transform=None, augment_real=False):
        super().__init__(data_folder, dataframe, transform)
        self.augment_real = augment_real
        self.real_indices = dataframe[dataframe['label'] == 1].index.tolist()
        self.fake_indices = dataframe[dataframe['label'] == 0].index.tolist()
        
        if augment_real:
            real_heavy_transform, _ = get_advanced_transforms(224, 224, is_real_heavy=True)
            self.real_heavy_transform = real_heavy_transform
    
    def __getitem__(self, idx):
        if self.augment_real and idx in self.real_indices and np.random.random() < 0.4:
            image = self.load_image(idx)
            if self.real_heavy_transform:
                image = self.real_heavy_transform(image=np.array(image))['image']
            return image, self.dataframe.iloc[idx]['label']
        else:
            return super().__getitem__(idx)

def extract_balanced_features(model, dataloader, device, max_samples=3000):
    model.eval()
    all_features = []
    all_labels = []
    samples_per_class = {0: 0, 1: 0}
    max_per_class = max_samples // 2
    
    with torch.no_grad():
        for images, labels in dataloader:
            if all(samples_per_class[k] >= max_per_class for k in samples_per_class):
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            keep_indices = []
            for i, label in enumerate(labels):
                label_val = int(label.item())
                if samples_per_class[label_val] < max_per_class:
                    keep_indices.append(i)
                    samples_per_class[label_val] += 1
            
            if not keep_indices:
                continue
            
            selected_images = images[keep_indices]
            selected_labels = labels[keep_indices]
            
            features = model.backbone(selected_images)
            all_features.append(features.cpu())
            all_labels.append(selected_labels.cpu())
    
    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"Balanced feature extraction: {features_tensor.shape[0]} samples")
    print(f"Class distribution: {torch.bincount(labels_tensor.long())}")
    
    return features_tensor, labels_tensor

def create_advanced_model(config, selected_features=None):
    model = BalancedQuantumClassifier(
        num_classes=config.classes,
        selected_features=selected_features,
        freeze_backbone=config.freeze_backbone,
        dropout=config.dropout
    ).to(DEVICE)
    
    criterion = AdaptiveFocalLoss(
        alpha=config.focal_alpha,
        gamma=config.focal_gamma,
        pos_weight=config.pos_weight,
        uncertainty_weight=config.uncertainty_weight,
        smoothing=config.label_smoothing
    )
    
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 
         'lr': config.learning_rate * 0.1, 'weight_decay': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'quantum_attention' in n], 
         'lr': config.learning_rate * 0.7, 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n and 'quantum_attention' not in n], 
         'lr': config.learning_rate, 'weight_decay': 5e-4}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=config.steps_per_epoch,
        epochs=config.epochs,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=100.0
    )
    
    return model, criterion, optimizer, scheduler

def compute_comprehensive_metrics(outputs, labels, confidences=None):
    with torch.no_grad():
        probabilities = torch.sigmoid(outputs)
        predictions = probabilities > 0.5
        labels_bool = labels.bool()
        
        correct = (predictions.squeeze() == labels_bool.squeeze()).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        
        tp = ((predictions.squeeze() == True) & (labels_bool.squeeze() == True)).sum().item()
        fp = ((predictions.squeeze() == True) & (labels_bool.squeeze() == False)).sum().item()
        fn = ((predictions.squeeze() == False) & (labels_bool.squeeze() == True)).sum().item()
        tn = ((predictions.squeeze() == False) & (labels_bool.squeeze() == False)).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        balanced_acc = balanced_accuracy_score(labels_bool.cpu().numpy(), predictions.squeeze().cpu().numpy())
        
        try:
            auc_roc = roc_auc_score(labels_bool.cpu().numpy(), probabilities.cpu().numpy())
            precision_curve, recall_curve, _ = precision_recall_curve(labels_bool.cpu().numpy(), probabilities.cpu().numpy())
            auc_pr = auc(recall_curve, precision_curve)
        except:
            auc_roc = 0.5
            auc_pr = 0.5
        
        metrics = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }
        
        if confidences is not None:
            correct_predictions = (predictions.squeeze() == labels_bool.squeeze()).float()
            confidence_scores = confidences.squeeze()
            ece = calculate_ece(confidence_scores.cpu().numpy(), correct_predictions.cpu().numpy())
            metrics['ece'] = ece
        
        return metrics

def calculate_ece(confidences, accuracies, n_bins=15):
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

def mixup_criterion(criterion, pred, y_a, y_b, lam, features=None, confidences=None):
    loss_a = criterion(pred, y_a, features, confidences)
    loss_b = criterion(pred, y_b, features, confidences)
    return lam * loss_a + (1 - lam) * loss_b

def train_epoch_advanced(model, dataloader, criterion, optimizer, scheduler, device, mixup_aug, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_metrics = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        
        if use_mixup and np.random.random() < 0.4:
            mixed_images, y_a, y_b, lam = mixup_aug(images, labels)
            
            optimizer.zero_grad()
            outputs, confidences = model(mixed_images, return_confidence=True)
            
            with torch.no_grad():
                features = model.backbone(mixed_images)
                if model.selected_features is not None:
                    features = features[:, model.selected_features]
            
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam, features, confidences)
        else:
            optimizer.zero_grad()
            outputs, confidences = model(images, return_confidence=True)
            
            with torch.no_grad():
                features = model.backbone(images)
                if model.selected_features is not None:
                    features = features[:, model.selected_features]
            
            loss = criterion(outputs, labels, features, confidences)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if not use_mixup or np.random.random() >= 0.4:
            metrics = compute_comprehensive_metrics(outputs, labels, confidences)
            all_metrics.append(metrics)
        
        running_loss += loss.item()
    
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    else:
        avg_metrics = {'accuracy': 0, 'f1_score': 0, 'auc_roc': 0.5}
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, avg_metrics

def validate_epoch_advanced(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs, confidences = model(images, return_confidence=True)
            
            features = model.backbone(images)
            if model.selected_features is not None:
                features = features[:, model.selected_features]
            
            loss = criterion(outputs, labels, features, confidences)
            
            metrics = compute_comprehensive_metrics(outputs, labels, confidences)
            all_metrics.append(metrics)
            running_loss += loss.item()
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, avg_metrics

def train_balanced_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, mixup_aug):
    best_f1 = 0.0
    best_balanced_acc = 0.0
    patience_counter = 0
    max_patience = 15
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_balanced_acc': [], 'val_balanced_acc': [],
        'train_f1': [], 'val_f1': [],
        'val_auc_roc': [], 'val_auc_pr': [], 'val_ece': []
    }
    
    for epoch in range(config.epochs):
        print(f"\n=== Epoch {epoch+1}/{config.epochs} ===")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.2e}")
        
        train_loss, train_metrics = train_epoch_advanced(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, mixup_aug, 
            use_mixup=epoch > 2
        )
        
        val_loss, val_metrics = validate_epoch_advanced(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics.get('accuracy', 0))
        history['val_acc'].append(val_metrics.get('accuracy', 0))
        history['train_balanced_acc'].append(train_metrics.get('balanced_accuracy', 0))
        history['val_balanced_acc'].append(val_metrics.get('balanced_accuracy', 0))
        history['train_f1'].append(train_metrics.get('f1_score', 0))
        history['val_f1'].append(val_metrics.get('f1_score', 0))
        history['val_auc_roc'].append(val_metrics.get('auc_roc', 0.5))
        history['val_auc_pr'].append(val_metrics.get('auc_pr', 0.5))
        history['val_ece'].append(val_metrics.get('ece', 0.5))
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics.get('accuracy', 0)*100:.2f}%, "
              f"Bal-Acc: {train_metrics.get('balanced_accuracy', 0)*100:.2f}%, F1: {train_metrics.get('f1_score', 0):.4f}")
        
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics.get('accuracy', 0)*100:.2f}%, "
              f"Bal-Acc: {val_metrics.get('balanced_accuracy', 0)*100:.2f}%, F1: {val_metrics.get('f1_score', 0):.4f}")
        
        print(f"      - AUC-ROC: {val_metrics.get('auc_roc', 0.5):.4f}, AUC-PR: {val_metrics.get('auc_pr', 0.5):.4f}, "
              f"ECE: {val_metrics.get('ece', 0.5):.4f}")
        
        print(f"      - Precision: {val_metrics.get('precision', 0):.4f}, "
              f"Recall: {val_metrics.get('recall', 0):.4f}, "
              f"Specificity: {val_metrics.get('specificity', 0):.4f}")
        
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_metrics.get('accuracy', 0) * 100,
            "val_accuracy": val_metrics.get('accuracy', 0) * 100,
            "train_balanced_accuracy": train_metrics.get('balanced_accuracy', 0) * 100,
            "val_balanced_accuracy": val_metrics.get('balanced_accuracy', 0) * 100,
            "train_f1": train_metrics.get('f1_score', 0),
            "val_f1": val_metrics.get('f1_score', 0),
            "val_precision": val_metrics.get('precision', 0),
            "val_recall": val_metrics.get('recall', 0),
            "val_specificity": val_metrics.get('specificity', 0),
            "val_auc_roc": val_metrics.get('auc_roc', 0.5),
            "val_auc_pr": val_metrics.get('auc_pr', 0.5),
            "val_ece": val_metrics.get('ece', 0.5),
            "learning_rate": current_lr
        }
        
        wandb.log(log_dict)
        
        current_f1 = val_metrics.get('f1_score', 0)
        current_balanced_acc = val_metrics.get('balanced_accuracy', 0)
        
        improved = False
        save_reason = ""
        
        if current_f1 > best_f1 + 0.005:
            best_f1 = current_f1
            improved = True
            save_reason = "best_f1"
        elif current_balanced_acc > best_balanced_acc + 0.005:
            best_balanced_acc = current_balanced_acc
            improved = True
            save_reason = "best_balanced_acc"
        elif abs(current_f1 - best_f1) < 0.01 and current_balanced_acc > best_balanced_acc:
            best_balanced_acc = current_balanced_acc
            improved = True
            save_reason = "best_balanced_acc_close"
        
        if improved:
            patience_counter = 0
            
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            
            save_path = os.path.join(CHECKPOINT_PATH, f"advanced_quantum_model_{save_reason}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
                'selected_features': model.selected_features,
                'config': dict(config),
                'history': history
            }, save_path)
            
            print(f"Model saved: {save_reason} (F1: {current_f1:.4f}, Bal-Acc: {current_balanced_acc:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping after {max_patience} epochs without improvement")
            break
    
    return history

def model_pipeline_balanced(hyperparameters):
    with wandb.init(project="balanced-quantum-deepfake-detection", config=hyperparameters):
        config = wandb.config
        
        dataset_folds = get_stratified_datasets(DATA_FOLDER, val_size=0.15, n_folds=1)
        train_df, val_df = dataset_folds[0]
        
        train_transform, val_transform = get_advanced_transforms(IMG_WIDTH, IMG_HEIGHT)
        
        train_dataset = BalancedFaceDataset(DATA_FOLDER, train_df, train_transform, augment_real=True)
        val_dataset = BalancedFaceDataset(DATA_FOLDER, val_df, val_transform, augment_real=False)
        
        train_sampler = create_balanced_sampler(train_df['label'].tolist(), oversample_minority=True)
        
        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        
        config.steps_per_epoch = len(train_loader)
        
        print("Creating initial model for feature extraction...")
        initial_model, _, _, _ = create_advanced_model(config)
        
        if config.use_quantum_feature_selection:
            print("Performing advanced quantum feature selection...")
            features, labels = extract_balanced_features(
                initial_model, train_loader, DEVICE, max_samples=4000
            )
            
            selected_features = perform_advanced_quantum_feature_selection(
                features, labels, num_select=config.num_selected_features
            )
            
            wandb.log({
                "total_features": features.shape[1],
                "selected_features": len(selected_features),
                "feature_reduction_ratio": len(selected_features) / features.shape[1]
            })
        else:
            selected_features = None
        
        print("Creating advanced balanced quantum model...")
        model, criterion, optimizer, scheduler = create_advanced_model(config, selected_features)
        model.selected_features = selected_features
        
        mixup_aug = MixupAugmentation(alpha=0.3)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        if selected_features:
            print(f"  Selected features: {len(selected_features)}/512 ({len(selected_features)/512*100:.1f}%)")
        
        wandb.log({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params
        })
        
        print("Starting balanced quantum training...")
        history = train_balanced_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, mixup_aug)
        
        print("\nTraining Summary:")
        print(f"  Best Validation F1: {max(history['val_f1']):.4f}")
        print(f"  Best Validation Balanced Accuracy: {max(history['val_balanced_acc']):.4f}")
        print(f"  Best Validation AUC-ROC: {max(history['val_auc_roc']):.4f}")
        print(f"  Final ECE: {history['val_ece'][-1] if history['val_ece'] else 'N/A'}")
    
    return model, history

if __name__ == "__main__":
    hyperparameters = dict(
        epochs=40,
        classes=1,
        learning_rate=0.0008,
        freeze_backbone=False,
        use_quantum_feature_selection=True,
        num_selected_features=450,
        dropout=0.4,
        focal_alpha=0.75,
        focal_gamma=2.5,
        pos_weight=8.0,
        uncertainty_weight=0.005,
        label_smoothing=0.1,
        dataset="Balanced Enhanced Faces Dataset",
        architecture="Advanced-BalancedQuantumInceptionResnetV1",
        augmentation_strategy="class_aware_heavy_real",
        sampling_strategy="weighted_oversampling",
        loss_function="adaptive_focal_with_uncertainty"
    )
    
    print(f"Running on device: {DEVICE.upper()}")
    print("Advanced Balanced Quantum Training Configuration:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    
    model, history = model_pipeline_balanced(hyperparameters)
    print("Advanced balanced quantum training completed!")