from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256), # [B, ]
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ) 

    def forward(self, x):
        """Forward pass of the model."""
        
        return self.model(x.flatten(1))