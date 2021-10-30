import numpy as np

from models.model import Model


class RidgePolyfitModel(Model):

    def __init__(
            self,
            x_vals: np.ndarray,
            y_vals: np.ndarray,
            deg: int,
            lam: float
            ):
        super().__init__(x_vals, y_vals)
        self.deg = deg
        self.lam = lam
        self.model: np.ndarray = np.zeros(self.deg + 1)

    def fit(self) -> None:
        X = np.zeros((self.x_vals.shape[0], self.deg+1))
        L = np.diag([0.0] + [self.lam] * self.deg)
        Y = self.y_vals.reshape(-1, 1)
        for i in range(self.deg+1):
            X[:, i] = self.x_vals ** i
        beta = np.linalg.inv(X.T @ X + L) @ X.T @ Y
        self.model = list(beta.flatten())

    def predict(self, x_vals: np.ndarray) -> np.ndarray:
        y_vals = np.zeros(len(x_vals))
        for index, parameter in enumerate(self.model):
            y_vals += parameter * x_vals ** index
        return y_vals
