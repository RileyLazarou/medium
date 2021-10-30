import os
from typing import Optional, Dict, List

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import seaborn as sns
from tqdm import tqdm

from models.polyfit_model import PolyfitModel
from models.ridge_polyfit_model import RidgePolyfitModel
import evaluate_model
from models.model import Model
import data_generator
import constants as C


sns.set()
N_POINTS = 8


def raw_data_intro():
    savedir = os.path.join("images", "raw", "intro2")
    os.makedirs(savedir, exist_ok=True)
    n = N_POINTS
    m = 5
    s=50
    points = data_generator.generate_data(n, "fixed")
    ylim = (-105, 205)
    xlim = (-0.5, 10.5)
    cmap = get_cmap("gnuplot2")

    # just points
    plt.figure()
    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=s,
        edgecolor='k',
        facecolor="none",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"scatter.png"))
    plt.close()

    predictions = []
    plt.figure()
    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=s,
        edgecolor='k',
        facecolor="none",
        zorder=60,
    )
    for deg in range(m):
        model = PolyfitModel(
            x_vals=points[:, 0],
            y_vals=points[:, 1],
            deg=deg
            )
        model.fit()
        _predictions = model.predict(evaluate_model.X_TEST)
        predictions.append(_predictions)
        mse = np.mean((model.predict(points[:, 0]) - points[:, 1])**2)
        plt.plot(
            evaluate_model.X_TEST,
            predictions[-1],
            label=f"Deg {deg}",
            c=cmap((1+deg)/(1+m)),
            zorder=50,
            )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    legend = plt.legend(loc='upper center', framealpha=1.0)
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))
    legend.set_zorder(100)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"poly.png"))
    plt.close()

    os.makedirs(savedir, exist_ok=True)
    model_type = PolyfitModel
    num_data = 8
    num_models = 100
    model_kwargs = {"deg": 2}
    predictions = evaluate_model.get_model_predictions(
        model_type=model_type,
        num_data=num_data,
        num_models=num_models,
        model_kwargs=model_kwargs,
    )

    ## Upper Plot: Many Models
    for i in range(predictions.shape[0]):
        label = "Models" if i == 0 else None
        plt.plot(
            evaluate_model.X_TEST,
            predictions[i, :],
            c='blue',
            alpha=0.8,
            linewidth=0.1,
            zorder=50,
            label=label,
        )
    plt.plot(
        evaluate_model.X_TEST,
        np.mean(predictions, axis=0),
        c='red',
        alpha=1,
        zorder=55,
        label="Average Model"
    )
    plt.plot(
        evaluate_model.X_TEST,
        evaluate_model.Y_TEST,
        c='k',
        alpha=1,
        zorder=60,
        label="Truth",
    )
    legend = plt.legend(loc='upper left', framealpha=1.0)
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))
    legend.set_zorder(100)
    plt.ylim(-55, 205)
    plt.xlim(-0.5, 10.5)
    plt.suptitle(f'Polynomial (Deg={2})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"combined.png"))
    plt.close()






def raw_data_scatterplot():
    savedir = os.path.join("images", "raw", "scatter")
    os.makedirs(savedir, exist_ok=True)
    print("Raw Data")
    counts = np.geomspace(8, 100000, 200)
    for index, i in tqdm(enumerate(counts), total=len(counts)):
        i = int(i)
        points = data_generator.generate_data(i, "fixed")
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=10,
            edgecolor='k',
            facecolor="none",
            )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.ylim(-105, 205)
        plt.xlim(-0.5, 10.5)
        plt.savefig(os.path.join(savedir, f"scatter.{i:06d}.{index}.png"))
        plt.close()



def make_single_model_plot(
        model_type: type(Model),
        num_data: int,
        num_models: int,
        model_kwargs: Optional[Dict] = None,
        ) -> None:
    # Collect Data
    predictions = evaluate_model.get_model_predictions(
        model_type=model_type,
        num_data=num_data,
        num_models=num_models,
        model_kwargs=model_kwargs,
        )
    bias, variance = evaluate_model.get_bias_and_variance(
        model_type=model_type,
        num_data=num_data,
        model_kwargs=model_kwargs
        )
    avg_squared_bias = np.mean(bias**2)
    avg_variance = np.mean(variance)

    # Make Plots
    fig, ax = plt.subplots(2, sharex=True, figsize=(8,6))

    ## Upper Plot: Many Models
    for i in range(predictions.shape[0]):
        label = "Models" if i == 0 else None
        ax[0].plot(
            evaluate_model.X_TEST,
            predictions[i, :],
            c='blue',
            alpha=0.8,
            linewidth=0.1,
            zorder=50,
            label=label,
            )
    ax[0].plot(
        evaluate_model.X_TEST,
        np.mean(predictions, axis=0),
        c='red',
        alpha=1,
        zorder=55,
        label="Average Model"
        )
    ax[0].plot(
        evaluate_model.X_TEST,
        evaluate_model.Y_TEST,
        c='k',
        alpha=1,
        zorder=60,
        label="Truth",
        )
    legend = ax[0].legend(loc='upper left', framealpha=1.0)
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))
    legend.set_zorder(100)
    ax[0].set_ylim(-55, 205)
    ax[0].set_xlim(-0.5, 10.5)

    # Bottom Plot: Bias and Variance
    ax[1].plot(
        evaluate_model.X_TEST,
        bias**2,
        label=f"bias² (avg={int(avg_squared_bias)})",
        c='red',
        zorder=50,
        )
    ax[1].plot(
        evaluate_model.X_TEST,
        variance,
        label=f"variance (avg={int(avg_variance)})",
        c='green',
        zorder=50,
        )
    ax[1].plot(
        evaluate_model.X_TEST,
        bias**2 + variance,
        label=f"error (avg={int(avg_squared_bias + avg_variance)})",
        c='blue',
        linestyle=":",
        linewidth=3,
        zorder=60,
        )
    ax[1].set_ylim(-50, 1200)
    legend = ax[1].legend(loc='upper left', framealpha=1.0)
    legend.set_zorder(100)
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))


def make_model_complexity_plot(
        model_type: type(Model),
        num_data: int,
        model_kwargs_list: List[Dict],
        ) -> None:
    bias_squareds = []
    variances = []
    for model_kwargs in model_kwargs_list:
        bias, variance = evaluate_model.get_bias_and_variance(
            model_type=model_type,
            num_data=num_data,
            model_kwargs=model_kwargs
            )
        bias_squareds.append(np.sum(bias**2))
        variances.append(np.sum(variance))
    errors = [x+y for x, y in zip(variances, bias_squareds)]
    plt.plot(
        np.arange(len(bias_squareds)),
        bias_squareds,
        c="red",
        label="bias²",
        zorder=50,
        )
    plt.plot(
        np.arange(len(variances)),
        variances,
        c="green",
        label="variance",
        zorder=50,
        )
    plt.plot(
        np.arange(len(errors)),
        errors,
        c="blue",
        label="error",
        zorder=60,
        linestyle=":",
        linewidth=3,
        )
    legend = plt.legend(loc='upper center', framealpha=1.0)
    legend.set_zorder(100)
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_facecolor((1, 1, 1, 0.8))


def polyfit_plots() -> None:
    # Run through Polyfit models of degree 0 to 9
    savedir = os.path.join("images", "performance", "polyfit")
    os.makedirs(savedir, exist_ok=True)
    model_type = PolyfitModel
    num_data = N_POINTS
    num_models = 100
    print("Polyfit:")
    for deg in tqdm(range(num_data), total=num_data):
        model_kwargs = {"deg": deg}
        make_single_model_plot(
            model_type=model_type,
            num_data=num_data,
            num_models=num_models,
            model_kwargs=model_kwargs,
        )
        plt.suptitle(f'Polynomial (Deg={deg})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, f"error.d{deg:02d}.png"))
        plt.close()

    # Plot Polyfit model complexity
    make_model_complexity_plot(
        model_type=model_type,
        num_data=num_data,
        model_kwargs_list=[{"deg": x} for x in range(num_data)],
    )
    plt.xticks(np.arange(num_data), np.arange(num_data))
    plt.xlabel("Polynomial Degree")
    plt.suptitle(f'Polynomial Degree Vs. Error', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"error.png"))
    plt.close()


def ridge_plots():
    # Run through Ridge Polyfit models of degree 0 to 9
    savedir = os.path.join("images", "performance", "ridge")
    os.makedirs(savedir, exist_ok=True)
    model_type = RidgePolyfitModel
    num_data = N_POINTS
    num_models = 100
    lam = 1
    print("ridge:")
    for deg in tqdm(range(num_data), total=num_data):
        model_kwargs = {"deg": deg, "lam": lam}
        make_single_model_plot(
            model_type=model_type,
            num_data=num_data,
            num_models=num_models,
            model_kwargs=model_kwargs,
        )
        plt.suptitle(f'Ridge Polynomial (Deg={deg}, lambda={lam})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, f"error.d{deg:02d}.png"))
        plt.close()

    # Plot Polyfit model complexity
    make_model_complexity_plot(
        model_type=model_type,
        num_data=num_data,
        model_kwargs_list=[{"deg": x, "lam": lam} for x in range(num_data)],
    )
    plt.xticks(np.arange(num_data), np.arange(num_data))
    plt.xlabel("Polynomial Degree")
    plt.suptitle(f'Ridge Polynomial Degree Vs. Error', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f"error.png"))
    plt.close()


def main():
    raw_data_intro()
    polyfit_plots()





if __name__ == "__main__":
    main()
