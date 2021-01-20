"""A script for training and visualizing MNIST AE training."""
from typing import List
import os
import shutil

import numpy as np
import torch
from torch import nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder


BATCH_SIZE = 32
STEPS = 500
LOG_DIR = 'AE_results'


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Make and return dataloaders for MNIST train and test data."""
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,))
        ])
    trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform)
    testset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform)
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True)
    test_loader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True)
    return train_loader, test_loader


def train_autoencoder_and_log(
        autoencoder: AutoEncoder,
        test_loader: DataLoader,
        ) -> None:
    """Train the autoencoder and log data."""
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.mkdir(LOG_DIR)
    train_losses = []
    test_losses = []

    AE = AutoEncoder(train_loader)
    train_losses = []
    for i in range(3):
        _losses = AE.train_step(32)
        train_losses.append(np.mean(_losses))
        print(i+1, train_losses[-1])
    my_iter = iter(test_loader)
    samples, __ = next(my_iter)
    reconstructed = AE.autoencode(samples).numpy()
    reconstructed = (reconstructed + 1) / 2
    reconstructed = reconstructed.transpose((0, 2, 3, 1))
    encoded = AE.encode(samples).numpy()
    for i in range(4):
        print(encoded[i])
        plt.figure()
        plt.imshow(reconstructed[i, :, :, 0])
        plt.figure()
        plt.imshow(((samples.numpy() + 1) / 2).transpose((0, 2, 3, 1))[i, :, :, 0])
        plt.show()
