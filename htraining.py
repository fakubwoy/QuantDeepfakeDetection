import os
import torch
import wandb
import pandas as pd
import albumentations
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# --- UPDATED: Import the FIXED Improved Hybrid Model ---
from models.cnn import ImprovedHybridQNN
from dataset.faces import FaceDataset

# --- Constants ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "./data"
TRAIN_BATCH_SIZE = 8  # Reduced batch size for quantum processing
VAL_BATCH_SIZE = 8
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_WORKERS = 2  # Reduced for stability
CHECKPOINT_PATH = "./models"

def get_datasets(data_folder, val_size=0.2):
    """
    Reads the faces.csv file and splits the data into training and validation sets.
    """
    faces_csv = os.path.join(data_folder, "faces", "faces.csv")
    faces_df = pd.read_csv(faces_csv)
    print(f"Original dataset length: {len(faces_df)}")
    
    train_df, val_df = train_test_split(faces_df, test_size=val_size, random_state=42, stratify=faces_df["label"])
    print(f"training dataset length ({1-val_size:.1%}): {len(train_df)}")
    print(f"validation dataset length ({val_size:.1%}): {len(val_df)}")
    
    return train_df, val_df

def get_transforms(img_width, img_height):
    """
    Defines the image augmentation and transformation pipelines.
    FIXED: Use the same transforms as classical model for consistency
    """
    train_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            p=0.5
        ),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    val_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])
    return train_transform, val_transform

def make_model(config):
    """
    Instantiates the model, optimizer, criterion, and scheduler.
    """
    model = ImprovedHybridQNN(
        num_qubits=4,  # Keep reduced for stability
        reps=1,        # Keep reduced for stability
        num_classes=config.classes,
        freeze_feature_extractor=True  # Keep feature extractor frozen
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # FIXED: Use a lower learning rate for hybrid models
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate * 0.1)  # 10x smaller LR
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)    
    return model, criterion, optimizer, scheduler

def calculate_accuracy(outputs, labels):
    """Calculate binary classification accuracy"""
    predicted = torch.sigmoid(outputs) > 0.5
    correct = (predicted == (labels > 0.5)).sum().item()
    total = labels.size(0)
    return correct / total

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """
    The main training loop.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        # Training phase
        for batch_idx, (images_batch, labels_batch) in enumerate(tqdm(train_loader, desc="Training")):
            images_batch = images_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # ADDED: Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels_batch)
        
        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader, desc="Validation"):
                images_batch = images_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE).float().view(-1, 1)
                
                outputs = model(images_batch)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels_batch)

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        
        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "val_loss": val_loss, 
            "train_accuracy": train_acc * 100,
            "val_accuracy": val_acc * 100, 
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # IMPROVED: Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            
            save_path = os.path.join(CHECKPOINT_PATH, f"hybrid_cnn.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, save_path)
            print(f"✅ New best model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {max_patience} epochs without improvement")
                break

def model_pipeline(hyperparameters):
    """
    Main pipeline to run the training process.
    """
    with wandb.init(project="deepfake-detection-improved-hybrid-fixed", config=hyperparameters):
        config = wandb.config
        print("Using the configuration from wandb.config")
        
        train_df, val_df = get_datasets(DATA_FOLDER)
        train_transform, val_transform = get_transforms(IMG_WIDTH, IMG_HEIGHT)
        
        train_dataset = FaceDataset(DATA_FOLDER, train_df, train_transform)
        val_dataset = FaceDataset(DATA_FOLDER, val_df, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        model, criterion, optimizer, scheduler = make_model(config)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("Beginning training 🎯")
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
    
    return model

if __name__ == "__main__":
    hyperparameters = dict(
        epochs=20,  # Increased epochs
        classes=1,
        learning_rate=0.001,  # Will be reduced 10x in make_model
        dataset="Faces Dataset",
        architecture="ImprovedHybrid-InceptionResnetV1-QNN-Fixed"
    )
    
    print(f"Running on device: {DEVICE.upper()}")
    model = model_pipeline(hyperparameters)