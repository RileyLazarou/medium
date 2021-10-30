"""Generate data for a given distribution."""
import numpy as np

import constants as C


def get_x_data(num: int, how: str = "fixed",) -> np.ndarray:
    if how == "fixed":
        x_vals = np.linspace(C.X_RANGE[0], C.X_RANGE[1], num)
    elif how == "random":
        x_vals = np.random.uniform(C.X_RANGE[0], C.X_RANGE[1], num)
    else:
        raise ValueError(f"Unexpected 'how' value '{how}'")
    return x_vals


def true_function(x_vals: np.ndarray) -> np.ndarray:
    """Given a numpy array of values, apply the true function without noise."""
    roots = (1, 4, 7)
    to_ret = np.ones(len(x_vals))
    for root in roots:
        to_ret *= (x_vals - root)
    return to_ret

# def true_function(x_vals: np.ndarray) -> np.ndarray:
#     """Given a numpy array of values, apply the true function without noise."""
#     to_ret = np.sin(x_vals**1.1)*50 + x_vals * 10
#     return to_ret


def generate_data(
        num: int,
        how: str = "fixed",
        ) -> np.ndarray:
    """Generate a 2D vector of sinusoidal + gaussian data points.

    Returned vector is of shape (num, 2), where each point is an x,y pair, with
    x in [0, 3pi) and y in R.
    Underlying distribution is y = sin(x) + eps, where eps ~ N(0, 0.4)

    Args:
        num: How many points to generate
        how: How to select x data. "fixed" to have them linearly spread between
            the upper and lower bound, "random" to have them selected randomly
            from U(0, 2pi).

    Returns:
        A numpy array of shape (num, 2) corresponding to num 2D points.
    """
    x_vals = get_x_data(num, how)
    y_vals = true_function(x_vals) + np.random.normal(0, C.STD, num)
    data = np.concatenate(
        (x_vals.reshape(-1, 1), y_vals.reshape(-1, 1)),
        axis=1)
    return data


