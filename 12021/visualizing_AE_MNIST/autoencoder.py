"""A simple, fully-connected AutoEncoder for MNIST digits."""
from typing import List

import numpy as np
import torch
from torch import nn
import torchvision as tv
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    """A simple, fully-connected encoder network for MNIST"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        intermediate = in_tensor.view((-1, 28*28))
        output_tensor = self.module(intermediate)

        return output_tensor


class Decoder(nn.Module):
    """A simple, fully-connected decoder network for MNIST"""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
            )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        intermediate = self.module(in_tensor)
        output_tensor = intermediate.view((-1, 1, 28, 28))

        return output_tensor


class AutoEncoder:
    """A simple, fully-connected AutoEncoder for MNIST digits."""

    def __init__(self, dataloader: DataLoader, latent_dim: int = 2) -> None:
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim).to(self.device)
        self.criterion = nn.L1Loss()
        self.optim = torch.optim.Adam(
            (*self.encoder.parameters(), *self.decoder.parameters()),
            lr=1e-3,)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an image as a latent vector."""
        image = image.to(self.device)
        with torch.no_grad():
            encoded = self.encoder(image)
        encoded = encoded.to("cpu")
        return encoded

    def autoencode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode then decode an image."""
        image = image.to(self.device)
        with torch.no_grad():
            encoded = self.encoder(image)
            decoded = self.decoder(encoded)
        decoded = decoded.to("cpu")
        return decoded

    def evaluate(self, testloader: DataLoader) -> int:
        """Return the average reconstruction loss of items in a dataloader."""
        cumulative_loss = 0
        count = 0
        for samples, __ in testloader:
            samples = samples.to(self.device)
            self.optim.zero_grad()
            encoded = self.encoder(samples)
            decoded = self.decoder(encoded)
            loss = self.criterion(samples, decoded)
            loss.backward()
            self.optim.step()
            cumulative_loss += loss.item()
            count += 1
        return cumulative_loss / count

    def train_step(self, steps: int = 1) -> List[float]:
        """Train for `steps` batches and return a list of losses."""
        losses = []
        while True:
            for samples, __ in self.dataloader:
                samples = samples.to(self.device)
                self.optim.zero_grad()
                encoded = self.encoder(samples)
                decoded = self.decoder(encoded)
                loss = self.criterion(samples, decoded)
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
                if len(losses) == steps:
                    return np.array(losses)


def main():
    import matplotlib.pyplot as plt
    batch_size = 32
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
        batch_size=batch_size,
        num_workers=2,
        shuffle=True)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=True)

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


if __name__ == '__main__':
    main()
