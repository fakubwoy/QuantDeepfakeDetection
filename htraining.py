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

# Import the new Hybrid Model
from models.cnn import HybridCNN
from dataset.faces import FaceDataset

# --- Constants ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_FOLDER = "./data"
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_WORKERS = 4
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
    """
    train_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_transform = albumentations.Compose([
        albumentations.Resize(img_width, img_height),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transform

def make_model(config):
    """
    Instantiates the model, optimizer, criterion, and scheduler.
    """
    model = HybridCNN(num_qubits=4, num_classes=config.classes).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Add a learning rate scheduler to reduce LR on plateau
# NEW Line (correct)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)    
    return model, criterion, optimizer, scheduler

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """
    The main training loop.
    """
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        
        for _, (images_batch, labels_batch) in enumerate(tqdm(train_loader)):
            images_batch = images_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images_batch.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader):
                images_batch = images_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE).float().view(-1, 1)
                
                outputs = model(images_batch)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * images_batch.size(0)
                
                predicted = torch.sigmoid(outputs) > 0.5
                total += labels_batch.size(0)
                correct += (predicted == (labels_batch > 0.5)).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Step the scheduler with the validation loss
        scheduler.step(val_epoch_loss)
        
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss, "val_loss": val_epoch_loss, "accuracy": accuracy, "learning_rate": optimizer.param_groups[0]['lr']})
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            
            save_path = os.path.join(CHECKPOINT_PATH, f"hybrid_cnn.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_path)
            print(f"Model saved to {save_path}")

def model_pipeline(hyperparameters):
    """
    Main pipeline to run the training process.
    """
    with wandb.init(project="deepfake-detection-hybrid", config=hyperparameters):
        config = wandb.config
        print("Using the configuration from wandb.config")
        
        train_df, val_df = get_datasets(DATA_FOLDER)
        train_transform, val_transform = get_transforms(IMG_WIDTH, IMG_HEIGHT)
        
        train_dataset = FaceDataset(DATA_FOLDER, train_df, train_transform)
        val_dataset = FaceDataset(DATA_FOLDER, val_df, val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        model, criterion, optimizer, scheduler = make_model(config)
        
        print("Beginning training 🎨")
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
    
    return model

if __name__ == "__main__":
    hyperparameters = dict(
        epochs=15, # Increased epochs to give scheduler time to work
        classes=1,
        learning_rate=0.001,
        dataset="Faces Dataset",
        architecture="HybridCNN-QNN-Trainable"
    )
    
    model = model_pipeline(hyperparameters)