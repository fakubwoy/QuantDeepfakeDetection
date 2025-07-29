# Quantum DeepFake Detection

The objective of this project is to distinguish between computer-generated (deepfake) and real individuals appearing in a video. However, it is important to note that this project is not optimized for identifying hyper-realistic generated images, but instead primarily focuses on analyzing screenshots taken from a video.

You can try the [deepfake detector](https://github.com/fakubwoy/QuantDeepfakeDetection) by uploading a screenshot from a deepfake video (for example from YouTube) and see the results.


## Getting Started 
 Clone the repository:
```sh
git clone https://github.com/fakubwoy/QuantDeepfakeDetection.git
```

## Project Structure 

```
deepfake-detection
│
└───dataset             # React component files
│   │   download.py     # Python script to download the dataset
│
data                    # The dataset will be stored here
│   │   ...
│
metrics                 # Metrics to be used in the model
│   │   ...
│
└───models              # Models to be used in the project
│   │   ...
│
└───saves               # Checkpoints & State Dicts of PyTorch
│   │   ...
│
└───utils               # Utility files (e.g. helper functions for visualization)
│   │   ...
│
│ training.py           # Python Script for training the model
│ training.ipynb        # Python Notebook for training the model
│
│ README.md
│ LICENSE  
│ gitignore  
```

## Set-Up Environment  
### Install the necessary dependencies
1. Create and activate a new conda environment:
```
conda create --name deepfake_env python=3.8
conda activate deepfake_env
```
2. Install dependencies using requirements.txt:

```
pip install -r requirements.txt
```

### Download the dataset
If you want to see the arguments that can be passed to download the dataset, run the following command:


1. Download the FaceForensics++ dataset:
```sh
python3 dataset/download.py data -c c40 -d DeepFakeDetection --server EU2
python3 dataset/download.py data -c c40 -d DeepFakeDetection_original --server EU2
```


## Download the Pretrained Model 

This project uses a pretrained classifier model (`resnetinceptionv1_epoch_32.pth`) for inference. Make sure to download this file and place it in the `models/` directory.

You can get the model file by running:

```bash
git lfs install
git lfs pull --include="models/resnetinceptionv1_epoch_32.pth"
```
If you have not cloned the repo with git lfs, you can install Git LFS and then pull the file manually. Otherwise, you'll see a pointer file instead of the actual model weights, which will result in loading errors.
## Train the model 
```sh
 python training.py
```

## Dataset

[FaceForensics](http://niessnerlab.org/projects/roessler2018faceforensics.html): A Large-scale Video Dataset for Forgery Detection in Human Faces.

[FakeCatcher](http://cs.binghamton.edu/~ncilsal2/DeepFakesDataset/): Dataset of synthesized images for deepfake detection.

[Kaggle Dataset augmented by Meta](https://ai.facebook.com/datasets/dfdc/): Dataset from the kaggle competition with more resources provided by Meta.

## TODO: Quantum Enhanced Deepfake Detection

This section outlines future research to explore how quantum algorithms can improve deepfake detection. Quantum Machine Learning (QML) could offer significant advantages due to its ability to process complex data in unique ways.

### Research & Implementation Areas:

* **Quantum Feature Detection:** Use quantum methods to find tiny, hidden patterns in deepfakes that classical methods might miss.
* **Hybrid Quantum-Classical Models:** Combine our existing PyTorch models with quantum components to boost accuracy and resilience.
* **Quantum Kernels:** Explore quantum techniques that make it easier to separate real from fake data, potentially leading to better classification.
* **Quantum GANs (QGANs):** Investigate using QGANs to understand how deepfakes are created (for defense) or even to generate new, harder-to-detect fakes for testing.
* **Quantum Frequency Analysis:** Apply quantum versions of signal processing to analyze video frames for deepfake artifacts in the frequency domain.
* **Quantum Optimization:** Use quantum algorithms to make our detection models train faster or perform better.
* **Quantum Tools & Platforms:** Experiment with quantum programming tools like Qiskit or PennyLane to test these ideas.

Ultimately, we aim for **more accurate, efficient, and robust deepfake detection** by leveraging the power of quantum computing.
