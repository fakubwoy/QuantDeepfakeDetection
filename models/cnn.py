import torch
import torch.nn as nn

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes # Added RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class HybridCNN(nn.Module):
    """
    Hybrid Quantum-Classical Convolutional Neural Network.
    """
    def __init__(self, num_qubits=4, num_classes=1):
        super(HybridCNN, self).__init__()
        
        # --- Classical Part ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.feature_reducer = nn.Linear(16 * 64 * 64, num_qubits)
        self.relu3 = nn.ReLU()

        # --- Quantum Part ---
        # 1. Encode the classical data using ZZFeatureMap
        feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
        
        # 2. Add a trainable quantum circuit (ansatz) to act as a neural network layer
        ansatz = RealAmplitudes(num_qubits, reps=1)
        
        # 3. Combine them into a single circuit
        qc = QuantumCircuit(num_qubits)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        # 4. Wrap the circuit in a QNN, specifying which parameters are for input and which are trainable weights
        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters  # These are now trainable weights
        )

        # 5. Wrap the QNN in a TorchConnector
        self.q_layer = TorchConnector(qnn)
        
        # Final classical layer to map the single quantum output to the desired class format
        self.final_fc = nn.Linear(1, num_classes)

    def forward(self, x):
        # --- Classical Feature Extraction ---
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        
        x = x.reshape(x.shape[0], -1)
        
        # --- Bottleneck ---
        x = self.relu3(self.feature_reducer(x))
        
        # --- Quantum Processing ---
        x = self.q_layer(x)
        
        # --- Final Classification ---
        x = self.final_fc(x)

        return x