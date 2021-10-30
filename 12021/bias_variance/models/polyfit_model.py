import numpy as np

from models.model import Model


class PolyfitModel(Model):

    def __init__(
            self,
            x_vals: np.ndarray,
            y_vals: np.ndarray,
            deg: int,
            ):
        super().__init__(x_vals, y_vals)
        self.deg = deg
        self.model: np.ndarray = np.zeros(self.deg + 1)

    def fit(self) -> None:
        self.model = np.polyfit(self.x_vals, self.y_vals, self.deg)

    def predict(self, x_vals: np.ndarray) -> np.ndarray:
        y_vals = np.zeros(len(x_vals))
        for index, parameter in enumerate(self.model[::-1]):
            y_vals += parameter * x_vals ** index
        return y_vals
