import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading face images.
    
    This dataset reads a CSV file containing image filenames and labels,
    loads the corresponding images, and applies specified transformations.
    """
    def __init__(self, data_folder, df, transform=None):
        """
        Args:
            data_folder (str): Path to the main data directory (e.g., './data').
            df (pd.DataFrame): DataFrame containing image names and labels.
            transform (albumentations.Compose, optional): Transformations to be applied.
        """
        self.data_folder = data_folder
        self.df = df
        self.transform = transform
        
        # Create a dictionary to map string labels to integer values
        self.label_map = {"real": 0, "fake": 1}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Fetches a single sample from the dataset.
        
        Args:
            index (int): The index of the sample.
            
        Returns:
            tuple: (image, label) where image is the transformed image tensor
                   and label is the integer label.
        """
        # Get the image filename and label from the DataFrame
        image_name = self.df.iloc[index]["name"]
        label_str = self.df.iloc[index]["label"]
        
        # Construct the full image path
        # The label (real/fake) is part of the path
        image_path = os.path.join(self.data_folder, "faces", label_str, image_name)
        
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            # Handle cases where an image might be missing or corrupt
            print(f"Warning: Could not read image {image_path}. Skipping.")
            # Return a placeholder or the next valid item
            return self.__getitem__((index + 1) % len(self))
            
        # Convert image from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations if they are provided
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        # Convert the string label to its integer equivalent
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        
        return image, label