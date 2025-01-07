import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WaterPlasticModel(nn.Module):
    def __init__(self, num_classes=2, in_channels=12):
        super(WaterPlasticModel, self).__init__()
        
        # Conv2D(16, 12, padding='same') -> kernel_size=12, padding=6 for same-like effect
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=12, padding=6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2D(32, 12, padding='same')
        self.conv2 = nn.Conv2d(16, 32, kernel_size=12, padding=6)
        
        # Conv2D(64, 12, padding='same')
        self.conv3 = nn.Conv2d(32, 64, kernel_size=12, padding=6)
        
        # We'll dynamically determine in_features for the first dense layer
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)
    
    def _initialize_fc1(self, x):
        """Dynamically initializes the first fully connected layer based on the flattened conv output."""
        in_features = x.view(x.size(0), -1).shape[1]
        self.fc1 = nn.Linear(in_features, 128).to(x.device)
    
    def forward(self, x):
        # Convolution + ReLU + Pool layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        if self.fc1 is None:
            self._initialize_fc1(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

