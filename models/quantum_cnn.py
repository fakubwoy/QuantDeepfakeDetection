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
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout

class AdvancedQuantumFeatureSelector:
    def __init__(self, num_features, num_select, p=2):
        self.num_features = num_features
        self.num_select = num_select
        self.p = p
        self.selected_features = None
        
    def create_enhanced_cost_hamiltonian(self, feature_scores, correlation_matrix):
        normalized_scores = 2 * (feature_scores - np.min(feature_scores)) / (np.max(feature_scores) - np.min(feature_scores)) - 1
        pauli_list = []
        
        for i, score in enumerate(normalized_scores):
            pauli_str = ['I'] * min(self.num_features, 16)
            if i < len(pauli_str):
                pauli_str[i] = 'Z'
                pauli_list.append((''.join(pauli_str), -score))
        
        for i in range(min(self.num_features, 16)):
            for j in range(i+1, min(self.num_features, 16)):
                if abs(correlation_matrix[i, j]) > 0.8:
                    pauli_str = ['I'] * min(self.num_features, 16)
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli_list.append((''.join(pauli_str), abs(correlation_matrix[i, j]) * 0.5))
        
        return SparsePauliOp.from_list(pauli_list) if pauli_list else None
    
    def hybrid_feature_selection(self, X, y):
        mi_scores = mutual_info_classif(X, y, random_state=42)
        f_scores, _ = f_classif(X, y)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        
        combined_scores = (mi_scores / np.max(mi_scores) + 
                          f_scores / np.max(f_scores) + 
                          rf_scores / np.max(rf_scores)) / 3
        
        return combined_scores
    
    def select_features(self, X, y):
        combined_scores = self.hybrid_feature_selection(X, y)
        
        if self.num_features <= 16:
            try:
                correlation_matrix = np.corrcoef(X.T)
                cost_hamiltonian = self.create_enhanced_cost_hamiltonian(combined_scores, correlation_matrix)
                
                if cost_hamiltonian is not None:
                    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=150), reps=self.p)
                    result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
                    best_bitstring = max(result.eigenstate, key=lambda x: abs(x)**2)
                    selected_indices = [i for i, bit in enumerate(best_bitstring) if abs(bit) > 0.3]
                    
                    if len(selected_indices) < self.num_select // 3:
                        selected_indices = np.argsort(combined_scores)[-self.num_select:].tolist()
                else:
                    selected_indices = np.argsort(combined_scores)[-self.num_select:].tolist()
                    
            except Exception as e:
                print(f"QAOA failed: {e}")
                selected_indices = np.argsort(combined_scores)[-self.num_select:].tolist()
        else:
            top_indices = np.argsort(combined_scores)[-self.num_select:]
            selected_indices = top_indices.tolist()
        
        self.selected_features = selected_indices
        return selected_indices


class MultiHeadQuantumAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.amplitude_real = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.05)
        self.amplitude_imag = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.05)
        self.phase_parameters = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        self.layer_norm = LayerNorm(feature_dim)
        self.dropout = Dropout(dropout)
        
    def quantum_superposition_weighting(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.num_heads, self.head_dim)
        
        amplitude_norm = torch.sqrt(self.amplitude_real**2 + self.amplitude_imag**2 + 1e-8)
        normalized_real = self.amplitude_real / amplitude_norm
        normalized_imag = self.amplitude_imag / amplitude_norm
        
        quantum_phase = torch.cos(self.phase_parameters) + 1j * torch.sin(self.phase_parameters)
        quantum_amplitude = (normalized_real + 1j * normalized_imag) * quantum_phase
        quantum_prob = torch.abs(quantum_amplitude)**2
        
        weighted_features = x_reshaped * quantum_prob.unsqueeze(0)
        return weighted_features.view(batch_size, self.feature_dim)
    
    def forward(self, x):
        residual = x
        x_quantum = self.quantum_superposition_weighting(x)
        
        Q = self.query(x_quantum)
        K = self.key(x_quantum)
        V = self.value(x_quantum)
        
        Q = Q.view(Q.size(0), self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(K.size(0), self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(V.size(0), self.num_heads, self.head_dim).transpose(0, 1)
        
        attention_weights = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.head_dim), dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended_features = attention_weights @ V
        attended_features = attended_features.transpose(0, 1).contiguous().view(-1, self.feature_dim)
        
        output = self.output_projection(attended_features)
        output = self.dropout(output)
        
        return self.layer_norm(output + residual)


class BalancedQuantumClassifier(nn.Module):
    def __init__(self, num_classes=1, selected_features=None, freeze_backbone=True, dropout=0.3):
        super().__init__()
        
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        self.feature_dim = 512
        self.selected_features = selected_features
        
        if selected_features is not None:
            self.effective_feature_dim = len(selected_features)
        else:
            self.effective_feature_dim = self.feature_dim
        
        self.feature_projection = nn.Sequential(
            nn.Linear(self.effective_feature_dim, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            LayerNorm(512)
        )
        
        self.quantum_attention = MultiHeadQuantumAttention(512, num_heads=8, dropout=dropout)
        
        self.auxiliary_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout//2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout//3),
            nn.Linear(128, num_classes)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(),
            nn.Dropout(dropout//2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_confidence=False):
        features = self.backbone(x)
        
        if self.selected_features is not None:
            features = features[:, self.selected_features]
        
        projected_features = self.feature_projection(features)
        attended_features = self.quantum_attention(projected_features)
        
        auxiliary_features = self.auxiliary_features(attended_features)
        combined_features = torch.cat([attended_features, auxiliary_features], dim=1)
        
        output = self.classifier(combined_features)
        
        if return_confidence:
            confidence = self.confidence_estimator(combined_features)
            return output, confidence
        
        return output


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0, pos_weight=10.0, uncertainty_weight=0.01, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.uncertainty_weight = uncertainty_weight
        self.smoothing = smoothing
        
    def forward(self, outputs, targets, features=None, confidences=None):
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        bce_loss = F.binary_cross_entropy_with_logits(
            outputs, targets_smooth, 
            pos_weight=torch.tensor(self.pos_weight).to(outputs.device),
            reduction='none'
        )
        
        probs = torch.sigmoid(outputs)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        total_loss = focal_loss.mean()
        
        if features is not None:
            feature_var = torch.var(features, dim=1, keepdim=True)
            uncertainty_reg = torch.mean(1.0 / (feature_var + 1e-6))
            total_loss += self.uncertainty_weight * uncertainty_reg
        
        if confidences is not None:
            confidence_loss = F.mse_loss(confidences, torch.abs(targets - probs).detach())
            total_loss += 0.1 * confidence_loss
        
        return total_loss


class MixupAugmentation:
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


def create_balanced_quantum_model(num_classes=1, num_selected_features=450, freeze_backbone=False):
    model = BalancedQuantumClassifier(
        num_classes=num_classes,
        selected_features=None,
        freeze_backbone=freeze_backbone,
        dropout=0.4
    )
    
    loss_fn = AdaptiveFocalLoss(
        alpha=0.75, 
        gamma=2.5, 
        pos_weight=8.0,
        uncertainty_weight=0.005,
        smoothing=0.1
    )
    
    return model, loss_fn


def perform_advanced_quantum_feature_selection(train_features, train_labels, num_select=450):
    selector = AdvancedQuantumFeatureSelector(
        num_features=train_features.shape[1],
        num_select=min(num_select, train_features.shape[1]),
        p=2
    )
    
    selected_indices = selector.select_features(train_features.cpu().numpy(), train_labels.cpu().numpy())
    
    print(f"Selected {len(selected_indices)} features using advanced quantum optimization")
    
    return selected_indices


if __name__ == "__main__":
    model, loss_fn = create_balanced_quantum_model(num_classes=1)
    
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    target = torch.randint(0, 2, (batch_size, 1)).float()
    
    features = model.backbone(x)
    loss = loss_fn(output, target, features)
    print(f"Loss: {loss.item()}")