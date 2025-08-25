import os
import torch
import torch.utils.data as data_utils
from torch import nn
from torch import optim
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb
# NEW IMPORTS for image transformation
import albumentations
from albumentations.pytorch import ToTensorV2
from utils.denormalize_image import denormalize_image
from utils.metrics import get_accuracy
from utils.dict_to_struct import DictStructure
from dataset.faces import FaceDataset

# --- Constants ---
DATA_ROOT = "data" 
FACES_CSV = os.path.join(DATA_ROOT, "faces", "faces.csv")
VALIDATION_SPLIT = 0.2
IMG_WIDTH = 224 # Standard size for InceptionResnetV1
IMG_HEIGHT = 224
CLASS_NAMES = {0: "real", 1: "fake"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_transforms(img_width, img_height):
    """
    Defines the image augmentation and transformation pipelines.
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


def make(config):
    """Create the dataloaders, the model, the criterion & the optimizer"""
    if isinstance(config, wandb.Config):
        print("Using the configuration from wandb.config")
    elif isinstance(config, dict):
        print("Not using the configuration from wandb.config")
        config = DictStructure(config)
    else:
        raise ValueError("config must be a wandb.Config or a dict")

    # --- Data Preparation ---
    faces_df = pd.read_csv(FACES_CSV)
    print(f"Original dataset length: {len(faces_df)}")

    train_df, val_df = train_test_split(
        faces_df, 
        test_size=VALIDATION_SPLIT, 
        random_state=42, 
        stratify=faces_df["label"]
    )
    print(f"Training dataset length: {len(train_df)}")
    print(f"Validation dataset length: {len(val_df)}")

    train_transform, val_transform = get_transforms(IMG_WIDTH, IMG_HEIGHT)
    
    train_dataset = FaceDataset(DATA_ROOT, train_df, transform=train_transform)
    val_dataset = FaceDataset(DATA_ROOT, val_df, transform=val_transform)

    train_loader = make_loader(train_dataset, batch_size=config.batch_size)
    val_loader = make_loader(val_dataset, batch_size=config.batch_size)
    
    # --- Model, Loss, and Optimizer ---
    model = InceptionResnetV1(
        pretrained=config.pretrained_dataset,
        classify=True,
        num_classes=config.classes,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # ADDED: Learning rate scheduler to prevent overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    return model, train_loader, val_loader, criterion, optimizer, scheduler


def make_loader(dataset, batch_size):
    """Create the Data Loader for the dataset."""
    loader = data_utils.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )
    return loader


def train_batch(images, labels, model, criterion, split="training", optimizer=None):
    """Train a batch of data."""
    if split == "training":
        assert optimizer is not None, "Optimizer must be provided for training"
    
    images = images.to(device=DEVICE)
    labels = labels.to(device=DEVICE).float()

    # Forward pass
    outputs = model(images).squeeze()

    loss = criterion(outputs, labels)
    acc = get_accuracy(outputs, labels)

    if split == "training":
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item(), acc


def model_pipeline(hyperparameters):
    """Initializes and runs the training pipeline."""
    with wandb.init(project="deepfake-detection", config=hyperparameters):
        config = wandb.config
        # UPDATED: Receive scheduler from make function
        model, train_loader, val_loader, criterion, optimizer, scheduler = make(config)
        # UPDATED: Pass scheduler to train function
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, config)
    return model


def save_best_model(model, optimizer, loss, save_path):
    """Saves the best model checkpoint, overwriting previous best."""
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": loss,
        },
        save_path,
    )


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    """Main training loop."""
    print("Beginning training 🎨")
    wandb.watch(model, criterion, log="all", log_freq=10)

    history = {"loss": {"train": [], "val": []}, "acc": {"train": [], "val": []}}
    batches_seen = 0
    # ADDED: Track best validation loss
    best_val_loss = float('inf')
    save_path = "./models/best_resnetinceptionv1.pth"

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        acc_per_epoch_train, loss_per_epoch_train = 0, 0
        for images_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"):
            loss, acc = train_batch(
                images_batch, labels_batch, model, criterion, "training", optimizer
            )
            batches_seen += 1
            acc_per_epoch_train += acc
            loss_per_epoch_train += loss

        # Validation phase
        model.eval()
        acc_per_epoch_val, loss_per_epoch_val = 0, 0
        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
                loss, acc = train_batch(
                    images_batch, labels_batch, model, criterion, "validation"
                )
                acc_per_epoch_val += acc
                loss_per_epoch_val += loss

        # Calculate and store metrics for the epoch
        train_loss = loss_per_epoch_train / len(train_loader)
        val_loss = loss_per_epoch_val / len(val_loader)
        train_acc = acc_per_epoch_train / len(train_loader)
        val_acc = acc_per_epoch_val / len(val_loader)
        
        history["loss"]["train"].append(train_loss)
        history["loss"]["val"].append(val_loss)
        history["acc"]["train"].append(train_acc)
        history["acc"]["val"].append(val_acc)

        # Print epoch results
        print(f"Epoch [{epoch + 1}/{config.epochs}]")
        print(f"\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"\tVal Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", end="\n\n")

        # UPDATED: Step the scheduler with the validation loss
        scheduler.step(val_loss)

        # Log metrics to wandb
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Train Acc": train_acc * 100,
            "Val Acc": val_acc * 100,
            "Learning Rate": optimizer.param_groups[0]['lr']
        }, step=batches_seen)

        # UPDATED: Save only the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(model, optimizer, best_val_loss, save_path)
            print(f"New best model saved to {save_path} with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    print(f"Running on device: {DEVICE.upper()}")

    # Hyperparameters for the training run
    config = dict(
        # UPDATED: Epochs set to 15
        epochs=15,
        classes=1,
        batch_size=16,
        learning_rate=0.001,
        dataset="FaceForensics++",
        architecture="InceptionResNetV1",
        pretrained_dataset="vggface2"
    )

    # Start the pipeline
    model = model_pipeline(config)
    
    print("Training finished.")
