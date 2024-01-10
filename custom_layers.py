import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

class FullyConnectedBlock(nn.Sequential):
    def __init__(self, input_size, output_size):
        super(FullyConnectedBlock, self).__init__(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Conv Blocks
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024)
        )
        
        # Fully Connected Blocks
        self.fc_blocks = nn.Sequential(
            nn.Flatten(),
            FullyConnectedBlock(9216, 1024),
            FullyConnectedBlock(1024, 1024),
            FullyConnectedBlock(1024, 1024),
            nn.Dropout(0.5)
        )
        
        # Output Layer
        self.output = nn.Linear(1024, 2)
        
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_blocks(x)
        x = self.output(x)
        return F.softmax(x, dim=1)
