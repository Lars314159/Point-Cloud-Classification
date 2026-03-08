import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_features, n_classes, n_neurons, n_hidden_layers):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, n_neurons, dtype=torch.float32), nn.ReLU(),
            *[nn.Linear(n_neurons, n_neurons, dtype=torch.float32), nn.ReLU()]*(n_hidden_layers-1),
            nn.Linear(n_neurons, n_classes, dtype=torch.float32)
        )
    def forward(self, x):
        return self.model.forward(x)
