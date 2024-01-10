import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class MaxPoolBlock(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPoolBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.max_pool(x)

class FullyConnectedBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super(FullyConnectedBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        return self.relu(self.fc(x))

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2, out_channels_3, out_channels_4):
        super(InceptionBlock, self).__init__()
        self.p1_1 = ConvBlock(in_channels, out_channels_1, kernel_size=1)
        self.p2_1 = ConvBlock(in_channels, out_channels_2[0], kernel_size=1)
        self.p2_2 = ConvBlock(out_channels_2[0], out_channels_2[1], kernel_size=3, padding=1)
        self.p3_1 = ConvBlock(in_channels, out_channels_3[0], kernel_size=1)
        self.p3_2 = ConvBlock(out_channels_3[0], out_channels_3[1], kernel_size=5, padding=2)
        self.p4_1 = MaxPoolBlock(kernel_size=3, stride=1)
        self.p4_2 = ConvBlock(in_channels, out_channels_4, kernel_size=1)

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return torch.cat((p1, p2, p3, p4), dim=1)

class MWTCNN(nn.Module):
    def __init__(self):
        super(MWTCNN, self).__init__()

        # Convolutional layers
        self.conv1 = ConvBlock(3, 32, kernel_size=3)
        self.conv2 = ConvBlock(32, 32, kernel_size=3)
        self.pool1 = MaxPoolBlock(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(32, 64, kernel_size=3)
        self.conv4 = ConvBlock(64, 64, kernel_size=3)
        self.conv5 = ConvBlock(64, 64, kernel_size=3)
        self.pool2 = MaxPoolBlock(kernel_size=2, stride=2)

        self.conv6 = ConvBlock(64, 128, kernel_size=3)
        self.conv7 = ConvBlock(128, 128, kernel_size=3)
        self.conv8 = ConvBlock(128, 128, kernel_size=3)
        self.pool3 = MaxPoolBlock(kernel_size=2, stride=2)

        self.conv9 = ConvBlock(128, 256, kernel_size=3)
        self.conv10 = ConvBlock(256, 256, kernel_size=3)
        self.conv11 = ConvBlock(256, 256, kernel_size=3)
        self.pool4 = MaxPoolBlock(kernel_size=2, stride=2)

        self.inception1 = InceptionBlock(256, 64, (96, 128), (16, 32), 32)
        self.inception2 = InceptionBlock(256, 128, (128, 192), (32, 96), 64)
        self.pool5 = MaxPoolBlock(kernel_size=3, stride=2, padding=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = FullyConnectedBlock(512 * 3 * 3, 1024)
        self.fc2 = FullyConnectedBlock(1024, 1024)
        self.fc3 = FullyConnectedBlock(1024, 1024)
        self.fc4 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=1)

        # Dropout layer
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.pool2(self.conv5(self.conv4(self.conv3(x))))
        x = self.pool3(self.conv8(self.conv7(self.conv6(x))))
        x = self.pool4(self.conv11(self.conv10(self.conv9(x))))
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool5(x)

        x = self.dropout(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        x = self.softmax(x)
        return x
