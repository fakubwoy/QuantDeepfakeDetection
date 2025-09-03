import torch
import torch.nn as nn

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridCNN(nn.Module):
    """
    Improved Hybrid Quantum-Classical Convolutional Neural Network.
    """
    def __init__(self, num_qubits=6, num_classes=1):  # bumped qubits: 4 -> 6
        super(HybridCNN, self).__init__()
        
        # --- Classical CNN Part ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # NEW: extra conv block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)

        # Feature reducer → match to num_qubits
        self.feature_reducer = nn.Linear(64 * 32 * 32, num_qubits)  # assumes input 256x256 → 32x32 after 3 pools
        self.relu_fc = nn.ReLU()

        # --- Quantum Part ---
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

        # Final classical layer
        self.final_fc = nn.Linear(1, num_classes)

    def forward(self, x):
        # --- Classical Feature Extraction ---
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))  # extra conv block
        x = self.dropout(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Reduce to quantum input size
        x = self.relu_fc(self.feature_reducer(x))

        # --- Quantum Layer ---
        x = self.q_layer(x)

        # --- Final Classification ---
        x = self.final_fc(x)
        return x
