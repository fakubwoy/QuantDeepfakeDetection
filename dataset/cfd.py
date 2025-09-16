import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

def setup_directories(base_data_folder):
    """Creates the necessary directories for storing face images."""
    faces_folder = os.path.join(base_data_folder, "faces")
    faces_real_folder = os.path.join(faces_folder, "real")
    faces_fake_folder = os.path.join(faces_folder, "fake")

    os.makedirs(faces_folder, exist_ok=True)
    os.makedirs(faces_real_folder, exist_ok=True)
    os.makedirs(faces_fake_folder, exist_ok=True)
    
    return faces_folder, faces_real_folder, faces_fake_folder

def get_video_paths(root_folder):
    """Recursively finds all .mp4 files in the given root folder."""
    video_paths = []
    for root, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".mp4"):
                video_paths.append(os.path.join(root, filename))
    return video_paths

def save_frame(frame_data, video_name, faces_folder, mtcnn_model):
    """Extracts a face from a single frame and saves it as an image."""
    try:
        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        face_tensor = mtcnn_model(frame_pil)

        if face_tensor is not None:
            face_np = face_tensor.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
            img_name = f"{os.path.splitext(video_name)[0]}_face.png"
            Image.fromarray(face_np).save(os.path.join(faces_folder, img_name))
    except Exception as e:
        print(f"Could not process frame from {video_name}: {e}")

def extract_and_save_faces(video_paths, output_folder, is_real, mtcnn_model):
    """Iterates through videos, extracts frames, and saves detected faces."""
    folder_tag = "real" if is_real else "fake"
    
    for video_path in tqdm(video_paths, desc=f"Extracting faces from {folder_tag} videos"):
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"Warning: Could not open video {video_path}")
            continue

        frame_skip = 60 if is_real else 90
        count = 0
        frame_num = 0
        
        ret, frame = video_capture.read()
        while ret:
            base_name = os.path.basename(video_path)
            frame_video_name = f"{os.path.splitext(base_name)[0]}_f{frame_num}.mp4"
            save_frame(frame, frame_video_name, output_folder, mtcnn_model)

            frame_num += 1
            count += frame_skip
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, count)
            ret, frame = video_capture.read()
        
        video_capture.release()

def create_csv_from_faces(faces_folder, real_folder, fake_folder):
    """Generates a CSV file listing all face images and their labels."""
    real_faces = [(name, "real") for name in os.listdir(real_folder)]
    fake_faces = [(name, "fake") for name in os.listdir(fake_folder)]

    all_faces = np.array(real_faces + fake_faces)
    np.random.shuffle(all_faces)

    df = pd.DataFrame(all_faces, columns=["name", "label"])
    csv_path = os.path.join(faces_folder, "faces.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSuccessfully created dataset CSV at: {csv_path}")
    print(f"Total real faces: {len(real_faces)}")
    print(f"Total fake faces: {len(fake_faces)}")


def main():
    """Main function to run the dataset creation process."""
    # --- Configuration ---
    BASE_DATA_FOLDER = "./data"
    FAKE_VIDEOS_ROOT = os.path.join(BASE_DATA_FOLDER, "manipulated_sequences")
    ORIGINAL_VIDEOS_ROOT = os.path.join(BASE_DATA_FOLDER, "original_sequences")
    
    # --- Setup ---
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device: {device}')
    
    mtcnn = MTCNN(
        select_largest=False, 
        post_process=False, 
        device=device
    ).eval()
    
    # --- Pipeline ---
    print("1. Setting up directories...")
    faces_folder, faces_real, faces_fake = setup_directories(BASE_DATA_FOLDER)
    
    print("2. Finding video files...")
    original_videos = get_video_paths(ORIGINAL_VIDEOS_ROOT)
    fake_videos = get_video_paths(FAKE_VIDEOS_ROOT)

    print(f"Found {len(original_videos)} real videos and {len(fake_videos)} fake videos.")

    print("\n3. Extracting faces from real videos...")
    extract_and_save_faces(original_videos, faces_real, is_real=True, mtcnn_model=mtcnn)
    
    print("\n4. Extracting faces from fake videos...")
    extract_and_save_faces(fake_videos, faces_fake, is_real=False, mtcnn_model=mtcnn)

    print("\n5. Creating final CSV file...")
    create_csv_from_faces(faces_folder, faces_real, faces_fake)

    print("\nDataset creation complete!")


if __name__ == '__main__':
    main()
