from typing import Optional, Dict, Tuple

import numpy as np

from models.model import Model
import data_generator
import constants as C


X_TEST = data_generator.get_x_data(C.NUM_PRESENTATION_DATA, how="fixed")
Y_TEST = data_generator.true_function(X_TEST)


def get_model_predictions(
        model_type: type(Model),
        num_data: int,
        num_models: int,
        model_kwargs: Optional[Dict] = None,
        ) -> np.ndarray:
    """Generate many model predictions.

    Args:
        model_type: The type of model to create
        num_data: The number of data points to generate
        num_models: The number of models to generate
        model_kwargs: Keyword arguments for building the model.

    Returns:
        predictions: A 2D numpy array of shape (num_models, num_data)
            where each row is the predictions for a different model

    """
    if not model_kwargs:
        model_kwargs = {}
    predictions = np.zeros((num_models, len(X_TEST)))
    for model_num in range(num_models):
        data = data_generator.generate_data(num_data)
        model = model_type(data[:, 0], data[:, 1], **model_kwargs)
        model.fit()
        y_hat = model.predict(X_TEST)
        predictions[model_num] = y_hat
    return predictions


def get_bias_and_variance(
        model_type: type(Model),
        num_data: int,
        model_kwargs: Optional[Dict] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return a model's bias and variances at various points."""
    predictions = get_model_predictions(
        model_type=model_type,
        num_data=num_data,
        num_models=1000,
        model_kwargs=model_kwargs,
        )
    y_vals_test = data_generator.true_function(X_TEST)
    mean_predictions = np.mean(predictions, axis=0)
    bias = mean_predictions - y_vals_test
    variance = np.mean((predictions - mean_predictions) ** 2, axis=0)
    return bias, variance
