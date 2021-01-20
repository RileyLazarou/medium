"""A script for training and visualizing MNIST AE training."""
from typing import Tuple
import os
import shutil

import numpy as np
import torch
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoencoder import AutoEncoder
import config


def get_data() -> Tuple[DataLoader, DataLoader]:
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
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        shuffle=False)
    test_loader = DataLoader(
        testset,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        shuffle=False)
    return train_loader, test_loader


def _get_sample_digits(loader: DataLoader) -> torch.Tensor:
    """Find and return one image of each digit, in order 0 to 9."""
    digits = {}
    for images, labels in loader:
        for image, label in zip(images, labels):
            label = int(label.item())
            if label not in digits:
                digits[label] = image
            if len(digits) == 10:
                break
        if len(digits) == 10:
            break
    images = torch.stack([digits[x] for x in range(10)])
    return images


def train_autoencoder_and_log(
        autoencoder: AutoEncoder,
        train_loader: DataLoader,
        test_loader: DataLoader,
        ) -> None:
    """Train the autoencoder and log data to disc."""
    if os.path.exists(config.LOG_DIR):
        shutil.rmtree(config.LOG_DIR)
    os.mkdir(config.LOG_DIR)
    train_losses = []
    test_losses = []
    test_images = _get_sample_digits(test_loader)
    all_encodings = []
    all_reconstructions = []
    for i in tqdm(range(config.STEPS)):
        # Train
        autoencoder.train_step(steps=1)

        # collect train and test losses
        train_losses.append(autoencoder.evaluate(train_loader))
        test_losses.append(autoencoder.evaluate(test_loader))

        # collect and save train encodings
        encodings = None
        for images, _ in train_loader:
            encodings_ = autoencoder.encode(images)
            if encodings is None:
                encodings = encodings_
            else:
                encodings = torch.cat((encodings, encodings_), 0)
        all_encodings.append(encodings)

        # collect the sample reconstructions
        reconstructions = autoencoder.autoencode(test_images)
        all_reconstructions.append(reconstructions)

    def save(filename, object):
        filename = os.path.join(config.LOG_DIR, filename)
        np.save(filename, object)



    # Format and save data
    train_losses = np.array(train_losses)
    save("train_losses.npy", train_losses)
    test_losses = np.array(test_losses)
    save("test_losses.npy", test_losses)

    test_images = (test_images.numpy() + 1) / 2
    save("test_images.npy", test_images)

    all_encodings = np.array([x.numpy() for x in all_encodings])
    save("encodings.npy", all_encodings)

    all_reconstructions = np.array([x.numpy() for x in all_reconstructions])
    all_reconstructions = (all_reconstructions + 1) / 2
    save("reconstructions.npy", all_reconstructions)

    labels = [x.numpy() for __, x in train_loader]
    labels = np.array(labels, dtype=int).flatten()
    save("labels.npy", labels)


def main():
    """Load data, build autoencoder, train, and log data."""
    train_loader, test_loader = get_data()
    autoencoder = AutoEncoder(train_loader, latent_dim=2)
    train_autoencoder_and_log(
        autoencoder=autoencoder,
        train_loader=train_loader,
        test_loader=test_loader,
        )


if __name__ == '__main__':
    main()
