"""A simple, fully-connected AutoEncoder for MNIST digits."""
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import config


class Encoder(nn.Module):
    """A simple, fully-connected encoder network for MNIST."""

    def __init__(self, latent_dim: int) -> None:
        """Initialize the encoder with a specified latent dim."""
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
    """A simple, fully-connected decoder network for MNIST."""

    def __init__(self, latent_dim: int) -> None:
        """Initialize the decoder with a specified latent dim."""
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
        """Initialize the autoencoder with a dataloader and latent dim."""
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(latent_dim).to(self.device)
        self.decoder = Decoder(latent_dim).to(self.device)
        self.criterion = nn.L1Loss()
        self.optim = torch.optim.Adam(
            (*self.encoder.parameters(), *self.decoder.parameters()),
            lr=config.LR,)

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
                    yield np.array(losses)
                    losses = []
