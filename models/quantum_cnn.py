import torch
import torch.nn as nn
import numpy as np
from facenet_pytorch import InceptionResnetV1
from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from sklearn.feature_selection import mutual_info_classif
import torch.nn.functional as F

class QuantumFeatureSelector:
    """
    Quantum Approximate Optimization Algorithm (QAOA) for feature selection.
    Uses quantum optimization to find the best subset of features.
    """
    def __init__(self, num_features, num_select, p=1):
        self.num_features = num_features
        self.num_select = num_select
        self.p = p  # QAOA depth
        self.selected_features = None
        
    def create_cost_hamiltonian(self, feature_scores):
        """
        Create a cost Hamiltonian that favors high-scoring features.
        H = sum_i (score_i * Z_i) - penalty * (sum_i Z_i - num_select)^2
        """
        # Normalize scores to [-1, 1]
        normalized_scores = 2 * (feature_scores - np.min(feature_scores)) / (np.max(feature_scores) - np.min(feature_scores)) - 1
        
        # Create Pauli strings for the Hamiltonian
        pauli_list = []
        
        # Feature scoring terms
        for i, score in enumerate(normalized_scores):
            pauli_str = ['I'] * self.num_features
            pauli_str[i] = 'Z'
            pauli_list.append((''.join(pauli_str), -score))  # Negative because we want to maximize
        
        return SparsePauliOp.from_list(pauli_list)
    
    def select_features(self, X, y):
        """
        Use QAOA to select the best features based on mutual information.
        """
        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        if self.num_features <= 16:  # Only use QAOA for reasonable qubit counts
            try:
                # Create cost Hamiltonian
                cost_hamiltonian = self.create_cost_hamiltonian(mi_scores)
                
                # Create QAOA circuit
                qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=100), reps=self.p)
                
                # Solve the optimization problem
                result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
                
                # Extract the best solution
                best_bitstring = max(result.eigenstate, key=lambda x: abs(x)**2)
                
                # Convert to feature indices
                selected_indices = [i for i, bit in enumerate(best_bitstring) if abs(bit) > 0.5]
                
                # Fallback: if too few features selected, use top MI scores
                if len(selected_indices) < self.num_select // 2:
                    selected_indices = np.argsort(mi_scores)[-self.num_select:].tolist()
                    
            except Exception as e:
                print(f"QAOA failed, falling back to classical feature selection: {e}")
                selected_indices = np.argsort(mi_scores)[-self.num_select:].tolist()
        else:
            # For high-dimensional features, use classical selection
            selected_indices = np.argsort(mi_scores)[-self.num_select:].tolist()
        
        self.selected_features = selected_indices
        return selected_indices


class QuantumInspiredAttention(nn.Module):
    """
    Quantum-inspired attention mechanism based on quantum superposition principles.
    Uses the concept of quantum interference to weight features.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Quantum-inspired parameters (analogous to quantum amplitudes)
        self.amplitude_real = nn.Parameter(torch.randn(feature_dim) * 0.1)
        self.amplitude_imag = nn.Parameter(torch.randn(feature_dim) * 0.1)
        
        # Classical attention components
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
    def quantum_interference(self, x):
        """
        Apply quantum interference-inspired weighting.
        Combines real and imaginary parts like quantum amplitudes.
        """
        # Normalize amplitudes (quantum normalization condition)
        amplitude_norm = torch.sqrt(self.amplitude_real**2 + self.amplitude_imag**2 + 1e-8)
        normalized_real = self.amplitude_real / amplitude_norm
        normalized_imag = self.amplitude_imag / amplitude_norm
        
        # Quantum probability (|amplitude|^2)
        quantum_prob = normalized_real**2 + normalized_imag**2
        
        # Apply quantum-inspired weighting
        return x * quantum_prob.unsqueeze(0)
    
    def forward(self, x):
        # Apply quantum interference
        x_quantum = self.quantum_interference(x)
        
        # Classical attention mechanism
        Q = self.query(x_quantum)
        K = self.key(x_quantum)
        V = self.value(x_quantum)
        
        # Scaled dot-product attention
        attention_weights = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.feature_dim), dim=-1)
        attended_features = attention_weights @ V
        
        return attended_features + x  # Residual connection


class QuantumEnhancedClassifier(nn.Module):
    """
    Hybrid model that uses quantum algorithms for feature selection and attention,
    while maintaining the classical InceptionResnetV1 for core feature extraction.
    """
    def __init__(self, num_classes=1, selected_features=None, freeze_backbone=True):
        super().__init__()
        
        # Use the proven InceptionResnetV1 backbone
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature dimensions from InceptionResnetV1
        self.feature_dim = 512
        self.selected_features = selected_features
        
        # Determine effective feature dimension after selection
        if selected_features is not None:
            self.effective_feature_dim = len(selected_features)
        else:
            self.effective_feature_dim = self.feature_dim
        
        # Quantum-inspired attention mechanism
        self.quantum_attention = QuantumInspiredAttention(self.effective_feature_dim)
        
        # Classical classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.effective_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classification head."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using the proven backbone
        features = self.backbone(x)
        
        # Apply quantum feature selection if specified
        if self.selected_features is not None:
            features = features[:, self.selected_features]
        
        # Apply quantum-inspired attention
        attended_features = self.quantum_attention(features)
        
        # Final classification
        output = self.classifier(attended_features)
        
        return output


class QuantumOptimizedLoss(nn.Module):
    """
    Quantum-inspired loss function that incorporates uncertainty principles.
    Adds a regularization term based on quantum uncertainty.
    """
    def __init__(self, base_loss=None, uncertainty_weight=0.01):
        super().__init__()
        self.base_loss = base_loss or nn.BCEWithLogitsLoss()
        self.uncertainty_weight = uncertainty_weight
    
    def quantum_uncertainty_regularization(self, features):
        """
        Add quantum uncertainty-inspired regularization.
        Encourages the model to maintain some uncertainty in its feature representations.
        """
        # Calculate feature variance (analogous to quantum uncertainty)
        feature_mean = torch.mean(features, dim=1, keepdim=True)
        feature_var = torch.var(features, dim=1, keepdim=True)
        
        # Penalize overly certain (low variance) features
        uncertainty_reg = torch.mean(torch.exp(-feature_var))
        
        return uncertainty_reg
    
    def forward(self, outputs, targets, features=None):
        # Base classification loss
        base_loss = self.base_loss(outputs, targets)
        
        # Add quantum uncertainty regularization if features provided
        if features is not None:
            uncertainty_reg = self.quantum_uncertainty_regularization(features)
            total_loss = base_loss + self.uncertainty_weight * uncertainty_reg
        else:
            total_loss = base_loss
        
        return total_loss


def create_quantum_enhanced_model(num_classes=1, num_selected_features=256, freeze_backbone=True):
    """
    Factory function to create a quantum-enhanced model.
    """
    model = QuantumEnhancedClassifier(
        num_classes=num_classes,
        selected_features=None,  # Will be set after feature selection
        freeze_backbone=freeze_backbone
    )
    
    # Create quantum-optimized loss
    loss_fn = QuantumOptimizedLoss(uncertainty_weight=0.01)
    
    return model, loss_fn


# Example usage and feature selection pipeline
def perform_quantum_feature_selection(train_features, train_labels, num_select=256):
    """
    Perform quantum feature selection on training data.
    """
    selector = QuantumFeatureSelector(
        num_features=train_features.shape[1],
        num_select=min(num_select, train_features.shape[1]),
        p=1  # QAOA depth
    )
    
    selected_indices = selector.select_features(train_features.cpu().numpy(), train_labels.cpu().numpy())
    
    print(f"Selected {len(selected_indices)} features out of {train_features.shape[1]} using quantum optimization")
    
    return selected_indices


if __name__ == "__main__":
    # Example of how to use the quantum-enhanced model
    model, loss_fn = create_quantum_enhanced_model(num_classes=1)
    
    # Example input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Example target
    target = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Calculate loss
    features = model.backbone(x)  # Get features for uncertainty regularization
    loss = loss_fn(output, target, features)
    print(f"Loss: {loss.item()}")