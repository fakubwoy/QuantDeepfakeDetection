import torch
import torch.nn as nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class ImprovedHybridQNN(nn.Module):
    """
    An improved Hybrid Quantum-Classical Network that:
    1. Uses a powerful, pre-trained InceptionResnetV1 as the feature extractor.
    2. Implements gradual feature reduction to avoid information loss.
    3. Utilizes a more expressive quantum circuit.
    """
    def __init__(self, num_qubits=8, reps=2, num_classes=1, freeze_feature_extractor=True):
        super(ImprovedHybridQNN, self).__init__()

        # --- 1. Use InceptionResnetV1 properly ---
        self.feature_extractor = InceptionResnetV1(pretrained='vggface2', classify=False)

        # Freeze the parameters of the feature extractor
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # --- 2. InceptionResnetV1 with classify=False outputs 512 features ---
        num_ftrs = 512

        # --- 3. Gradual Feature Reduction ---
        self.feature_reducer = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_qubits),
            nn.Tanh()
        )
        
        # --- 4. More Expressive Quantum Part ---
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=reps)
        ansatz = RealAmplitudes(num_qubits, reps=reps)

        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )

        self.q_layer = TorchConnector(qnn)

        # Final classical layer to map QNN output to the number of classes
        self.final_fc = nn.Linear(1, num_classes)

    def forward(self, x):
        # --- FIXED: Use the feature extractor without torch.no_grad() ---
        # The layers are already frozen, so gradients won't be computed for them
        # during training. Removing the no_grad() context allows tools like
        # Grad-CAM to function correctly during evaluation.
        x = self.feature_extractor(x)

        # --- Gradual Reduction ---
        x = self.feature_reducer(x)

        # --- Quantum Processing ---
        x = self.q_layer(x)

        # --- Final Classification ---
        x = self.final_fc(x)
        return x


# Keep the old class for reference
class HybridResnetQNN(nn.Module):
    """
    Hybrid Quantum-Classical Network using a pre-trained ResNet18 as a feature extractor.
    """
    def __init__(self, num_qubits=4, num_classes=1, freeze_feature_extractor=True):
        super(HybridResnetQNN, self).__init__()
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_qubits)
        self.relu = nn.ReLU()
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        ansatz = RealAmplitudes(num_qubits, reps=1)
        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters
        )
        self.q_layer = TorchConnector(qnn)
        self.final_fc = nn.Linear(1, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.relu(x)
        x = self.q_layer(x)
        x = self.final_fc(x)
        return x