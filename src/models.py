"""
models.py - Model Architectures for Investigating Simplicity Bias

Defines neural network architectures parameterized for systematic comparison:
  1. StandardMLP: Canonical 2-hidden layer MLP with ReLU non-linearities.
  2. DeepMLP: 4-hidden layer MLP to test depth and representational expressivity.
  3. ResMLP: Residual MLP with skip connections to evaluate optimization accessibility.
"""

import torch
import torch.nn as nn

class StandardMLP(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class DeepMLP(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class ResMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.act2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act1(self.fc1(x))
        out = self.fc2(out)
        return self.act2(out + residual)

class ResMLP(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU()
        )
        self.block1 = ResMLPBlock(hidden_dim)
        self.block2 = ResMLPBlock(hidden_dim)
        self.out_head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.out_head(h)

def create_model(arch_name: str, in_features: int = 8, hidden_dim: int = 64):
    name = arch_name.lower()
    if name in ["mlp", "standardmlp", "standard"]:
        return StandardMLP(in_features=in_features, hidden_dim=hidden_dim)
    elif name in ["deep", "deepmlp"]:
        return DeepMLP(in_features=in_features, hidden_dim=hidden_dim)
    elif name in ["res", "resmlp"]:
        return ResMLP(in_features=in_features, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
