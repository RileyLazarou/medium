import os
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import config


DATA_OBJECTS = ["train_losses", "test_losses", "test_images", "encodings",
                "reconstructions", "labels"]
CMAP = cm.get_cmap(config.CMAP)


def load_and_format_data() -> Dict[str, np.ndarray]:
    """Load and return training log data."""
    def load(obj):
        return np.load(os.path.join(config.LOG_DIR, f"{obj}.npy"))

    data = {x: load(x) for x in DATA_OBJECTS}
    return data


def visualize_step(data: Dict, step: int) -> None:
    fig = plt.figure(constrained_layout=False, figsize=(6, 4))
    gs1 = fig.add_gridspec(**config.GRID_SPEC_PARAMS)
    ax_encoding = fig.add_subplot(gs1[:-1, :-1])
    ax_loss = fig.add_subplot(gs1[-1, :-1])
    ax_digits = fig.add_subplot(gs1[:, -1])

    #ax_encoding.tick_params(direction="in")
    ax_digits.axis("off")

    # encodings
    legend = [Line2D([0], [0], color=CMAP(x/10), lw=4) for x in range(10)]
    colours = [CMAP(x/10) for x in data["labels"]]
    N = 5000
    ax_encoding.scatter(
        data["encodings"][step, :N, 0],
        data["encodings"][step, :N, 1],
        edgecolor=colours[:N],
        facecolor="none",
        s=1,
        )
    ax_encoding.legend(legend, list(range(10)), loc='upper right')
    ax_encoding.set_xlim(
        1.02*np.min(data["encodings"][..., 0]),
        1.02*np.max(data["encodings"][..., 0]),
        )
    ax_encoding.set_ylim(
        1.02*np.min(data["encodings"][..., 1]),
        1.02*np.max(data["encodings"][..., 1]),
        )

    # loss
    ax_loss.plot(
        np.arange(step+1),
        data["train_losses"][:step+1],
        label="Train MAE",
        )
    ax_loss.plot(
        np.arange(step+1),
        data["test_losses"][:step+1],
        label="Test MAE",
        )
    ax_loss.set_xlim(0, len(data["test_losses"]))
    ax_loss.set_ylim(
        0.98 * min(min(data["train_losses"]), min(data["test_losses"])),
        1.02 * max(max(data["train_losses"]), max(data["test_losses"])))
    ax_loss.legend(loc='upper right')

    # digits
    ims = np.zeros((10*28, 2*28))
    for index, image in enumerate(data["test_images"]):
        ims[index*28:(index+1)*28, :28] = image
    for index, image in enumerate(data["reconstructions"][step]):
        ims[index*28:(index+1)*28, 28:] = image
    ax_digits.imshow(ims, cmap="Greys")


data = load_and_format_data()
steps = len(data["train_losses"])
for i in range(steps):
    visualize_step(data, i)
    plt.savefig(os.path.join(config.LOG_DIR, f"figure.{i:05d}.png"))
