import torch
import torchvision
from augmentations import augmentation, ContrastiveAugmentation
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class initialize_dataset:
    def __init__(self, image_resolution=224, batch_size=128, MNIST=True):
        self.image_resolution= image_resolution
        self.batch_size=batch_size
        # self.MNIST = MNIST

    def load_dataset(self, data_dir= "/home/whitewalker/Documents/Vessel/classification/data", batch_size=32, test_split=0.2, num_workers=4):
    # Define data transformations
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the dataset
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # Calculate the size of training and test sets
        num_samples = len(dataset)
        num_test_samples = int(test_split * num_samples)
        num_train_samples = num_samples - num_test_samples

        # Split the dataset into training and test sets
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])

        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Create DataLoader for test set
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_dataloader, test_dataloader
