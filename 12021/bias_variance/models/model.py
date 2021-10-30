from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):

    def __init__(self, x_vals: np.ndarray, y_vals: np.ndarray):
        self.x_vals = x_vals
        self.y_vals = y_vals

    @abstractmethod
    def fit(self) -> None:
        """Fit the model using supplied training data."""
        pass

    @abstractmethod
    def predict(self, x_vals: np.ndarray) -> np.ndarray:
        """Given n feature values, return n predictions."""
        pass
