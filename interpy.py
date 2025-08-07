import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

use_checkpoint = True

if use_checkpoint:
    checkpoint = torch.load('./models/resnetinceptionv1_epoch_32.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(torch.load('./models/resnetinceptionv1_final.pth'))
model.to(DEVICE)
model.eval()


EXAMPLES_FOLDER = 'examples'
examples_names = os.listdir(EXAMPLES_FOLDER)
examples = []
for example_name in examples_names:
    example_path = os.path.join(EXAMPLES_FOLDER, example_name)
    label = example_name.split('_')[0]
    example = {
        'path': example_path,
        'label': label
    }
    examples.append(example)
np.random.shuffle(examples) 


example = examples[22]
example_img = example['path']
input_image = Image.open(example_img).convert('RGB')

input_tensor = torch.from_numpy(np.array(input_image))
face = mtcnn(input_tensor)

if face is None:
    raise Exception('No face detected')

face = face.unsqueeze(0) 
face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

face = face.to(torch.float32) / 255.0

target_layers = [model.block8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(0)]

grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
grayscale_cam = grayscale_cam[0, :]

face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
face_image_to_plot = (face_image_to_plot - face_image_to_plot.min()) / (face_image_to_plot.max() - face_image_to_plot.min())
face_image_to_plot_uint8 = (face_image_to_plot * 255).astype(np.uint8)

visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)

face_with_mask = cv2.addWeighted(face_image_to_plot_uint8, 1, visualization, 0.5, 0)


def predict(input_image:Image.Image, true_label:str):
    """
    Predict the label of the input_image and generate a GradCAM visualization.

    Args:
    input_image (Image.Image): The input image from which the face is to be detected and predicted.
    true_label (str): The true label of the input image for reference.

    Returns:
    tuple: A tuple containing:
        - confidences (dict): A dictionary with prediction confidences for 'real' and 'fake' labels.
        - true_label (str): The true label provided as input.
        - face_with_cam_overlay (Image.Image or None): An image with GradCAM overlay on the detected face, 
          or None if no face is detected.
    """

    face = mtcnn(input_image)
    if face is None:
        return {'real': 0.0, 'fake': 0.0}, "No face detected!", None 
    
    face = face.unsqueeze(0) 
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    face_image_for_display = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    face_image_for_display = Image.fromarray(face_image_for_display.astype(np.uint8))

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))

        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }

    target_layers = [model.block8]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)] 

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    face_np = np.array(face_image_for_display)
    face_np_normalized = (face_np - face_np.min()) / (face_np.max() - face_np.min())
    face_np_normalized_uint8 = (face_np_normalized * 255).astype(np.uint8)

    visualization = show_cam_on_image(face_np_normalized, grayscale_cam, use_rgb=True)
    face_with_cam_overlay = Image.fromarray(cv2.addWeighted(face_np_normalized_uint8, 1, visualization, 0.5, 0))

    return confidences, true_label, face_with_cam_overlay


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(label="Input Image", type="pil"),
        gr.Textbox(label="True Label (for testing, optional)", value="unknown") # Made optional for general use
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="True Label Provided"),
        gr.Image(label="Detected Face with CAM", type="pil") 
    ],
)

interface.launch()