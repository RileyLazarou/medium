import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os
import colorsys
from scipy.stats import norm

NUM_BATCHES = 60000
BATCH_SIZE = 512
PLOT_EVERY = 100
GRID_RESOLUTION = 400
FILE = ".".join(os.path.basename(__file__).split(".")[:-1])


def colourize(Z):
    c = [
        colorsys.hsv_to_rgb(x * 0.9, 1, y * 0.8 + 0.2)
        for (x, y) in (Z + 1) / 2
    ]
    return c


def target_function(Z):
    """
    Map Z (U(-1, 1), U(-1, 1)) to (N(0, 1), N(0, 1))
    """
    return norm.ppf((Z + 1) / 2)


def generate_noise(samples):
    """
    Generate `samples` samples of uniform noise
    """
    noise = np.random.uniform(-1, 1, (samples, 2))
    return noise


def sample_from_target_function(samples):
    """
    sample from the target function
    """
    Z = generate_noise(samples)
    return target_function(Z)


def build_generator():
    """
    Build a generator mapping (N, N, onehot(8)) to ([-1,1], [-1,1])
    """
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(4):
        X = layers.Dense(16)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(2)(X)
    G = Model(input_layer, output_layer)
    return G


def build_discriminator():
    """
    Build a discriminator mapping (R, R) to [0, 1]
    """
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(512)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(1, activation="sigmoid")(X)
    D = Model(input_layer, output_layer)
    D.compile(
        Adam(learning_rate=0.001, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return D


def build_GAN(G, D):
    """
    Given a generator and a discriminator, build a GAN
    """
    D.trainable = False
    input_layer = layers.Input((2,))
    X = G(input_layer)
    output_layer = D(X)
    GAN = Model(input_layer, output_layer)
    GAN.compile(
        Adam(learning_rate=0.0002, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return GAN


grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
grid[:, :, 1] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
flat_grid = grid.reshape((-1, 2))
test_samples = sample_from_target_function(5000)
test_noise = generate_noise(5000)


image_dir = os.path.join("ims")
script_dir = os.path.join("ims", FILE)
target_function_dir = os.path.join("ims", FILE, "1")
gan_training_dir = os.path.join("ims", FILE, "2")
gan_function_dir = os.path.join("ims", FILE, "3")

## Set up directories
image_dir = os.path.join("ims")
script_dir = os.path.join(image_dir, FILE)
target_dir = os.path.join(script_dir, "1_target")
target_still_dir = os.path.join(target_dir, "1_still")
target_animation_dir = os.path.join(target_dir, "2_animation")
training_dir = os.path.join(script_dir, "2_training")
training_animation_dir = os.path.join(training_dir, "1_animation")
training_still_dir = os.path.join(training_dir, "2_still")
mapping_dir = os.path.join(script_dir, "3_mapping")
mapping_still_dir = os.path.join(mapping_dir, "1_still")
mapping_animation_dir = os.path.join(mapping_dir, "2_animation")

for i in [
    image_dir,
    script_dir,
    target_dir,
    target_still_dir,
    target_animation_dir,
    training_dir,
    training_animation_dir,
    training_still_dir,
    mapping_dir,
    mapping_still_dir,
    mapping_animation_dir,
]:
    if not os.path.exists(i):
        os.makedirs(i)


grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
grid[:, :, 1] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
flat_grid = grid.reshape((-1, 2))
test_samples = sample_from_target_function(5000)
test_noise = generate_noise(5000)


def plot(
    G,
    D,
    GAN,
    step,
    step_count,
    D_accuracy,
    D_loss,
    G_accuracy,
    G_loss,
    filename,
):
    """
    Plots for the GAN gif
    """
    fake_samples = G.predict(test_noise, batch_size=len(test_noise))
    c = colourize(test_noise)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        fake_samples[:, 0],
        fake_samples[:, 1],
        edgecolor=c,
        facecolor="None",
        s=5,
        alpha=1,
        linewidth=1,
        zorder=50,
    )
    plt.scatter(
        test_samples[:, 0],
        test_samples[:, 1],
        edgecolor="#AAAAAA",
        facecolor="None",
        s=5,
        alpha=0.8,
        linewidth=1,
        zorder=30,
    )
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


## Part 1: Visualize the problem
### 1a: Input and output distributions
"""
lim = 3
f, ax = plt.subplots(1, 2, figsize=(12, 6))
input_points = generate_noise(10000)
output_points = target_function(input_points)
c = colourize(input_points)
ax[0].scatter(
    input_points[:, 0],
    input_points[:, 1],
    edgecolor=c,
    facecolor="None",
    s=5,
    alpha=1,
    linewidth=1,
)
ax[0].set_xlim(-lim, lim)
ax[0].set_ylim(-lim, lim)
ax[0].set_aspect(1)
ax[1].scatter(
    output_points[:, 0],
    output_points[:, 1],
    edgecolor=c,
    facecolor="None",
    s=5,
    alpha=1,
    linewidth=1,
)
ax[1].set_xlim(-lim, lim)
ax[1].set_ylim(-lim, lim)
ax[1].set_aspect(1)
plt.tight_layout()
plt.savefig(os.path.join(target_still_dir, "target.png"))
plt.close()


### 1b: 1a, but as a gif
steps = 300
space = np.linspace(input_points[:, :2], output_points, steps)
for step in range(steps):
    c = colourize(input_points)
    print(f"1b: {step:03d}/{steps}", end="\r")
    plt.figure(figsize=(6, 6))
    plt.scatter(
        space[step, :, 0],
        space[step, :, 1],
        edgecolor=c,
        facecolor="None",
        s=5,
        alpha=1,
        linewidth=1,
    )
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(target_animation_dir, f"target.{step:03d}.png"))
    plt.close()
print()
os.system(
    f"ffmpeg -r 20 -i {os.path.join(target_animation_dir, 'target.%03d.png')} "
    f"-crf 15 {os.path.join(target_animation_dir, 'target.mp4')}"
)
"""
## Part 2: GAN
if os.path.exists(os.path.join(mapping_dir, "G.h5")):
    G = load_model(os.path.join(mapping_dir, "G.h5"))
    D = load_model(os.path.join(mapping_dir, "D.h5"))
    GAN = build_GAN(G, D)
    skip_training = True
else:
    G = build_generator()
    D = build_discriminator()
    GAN = build_GAN(G, D)
    skip_training = False

step_count = []
D_accuracy = []
G_accuracy = []
D_loss = []
G_loss = []
count = 0
for step in range(NUM_BATCHES):
    if skip_training:
        break
    if step % 10 == 0:
        print(f"step {step}/{NUM_BATCHES}", end="\r")
    # Train discriminator
    D.trainable = True
    real_data = sample_from_target_function(BATCH_SIZE // 2)
    fake_data = G.predict(
        generate_noise(BATCH_SIZE // 2), batch_size=BATCH_SIZE // 2
    )
    data = np.concatenate((real_data, fake_data), axis=0)
    real_labels = np.ones((BATCH_SIZE // 2, 1))
    fake_labels = np.zeros((BATCH_SIZE // 2, 1))
    labels = np.concatenate((real_labels, fake_labels), axis=0)
    _D_loss, _D_accuracy = D.train_on_batch(data, labels)

    # Train generator
    D.trainable = False
    noise = generate_noise(BATCH_SIZE)
    labels = np.ones((BATCH_SIZE, 1))
    _G_loss, _G_accuracy = GAN.train_on_batch(noise, labels)

    if step % PLOT_EVERY == 0:
        step_count.append(step)
        D_loss.append(_D_loss)
        D_accuracy.append(_D_accuracy)
        G_loss.append(_G_loss)
        G_accuracy.append(_G_accuracy)
        plot(
            G=G,
            D=D,
            GAN=GAN,
            step=step,
            step_count=step_count,
            D_accuracy=D_accuracy,
            D_loss=D_loss,
            G_accuracy=G_accuracy,
            G_loss=G_loss,
            filename=os.path.join(training_animation_dir, f"{count:06d}.png"),
        )
        count += 1
else:
    print()
    G.save(os.path.join(mapping_dir, "G.h5"))
    D.save(os.path.join(mapping_dir, "D.h5"))
    os.system(
        f"ffmpeg -r 20 -i {os.path.join(training_animation_dir,'%06d.png')}"
        f" -crf 15 {os.path.join(training_animation_dir, 'training.mp4')}"
    )


## Part 1: Visualize the problem
### 1a: Input and output distributions
f, ax = plt.subplots(1, 2, figsize=(12, 6))
input_points = generate_noise(10000)
output_points = G.predict(input_points)
c = colourize(input_points)
ax[0].scatter(
    input_points[:, 0],
    input_points[:, 1],
    edgecolor=c,
    facecolor="None",
    s=5,
    alpha=1,
    linewidth=1,
)
ax[0].set_xlim(-3, 3)
ax[0].set_ylim(-3, 3)
ax[0].set_aspect(1)
ax[1].scatter(
    output_points[:, 0],
    output_points[:, 1],
    edgecolor=c,
    facecolor="None",
    s=5,
    alpha=1,
    linewidth=1,
)
ax[1].set_xlim(-3, 3)
ax[1].set_ylim(-3, 3)
ax[1].set_aspect(1)
plt.tight_layout()
plt.savefig(os.path.join(mapping_still_dir, "mapping.png"))
plt.close()


### 1b: 1a, but as a gif
steps = 300
space = np.linspace(input_points[:, :2], output_points, steps)
for step in range(steps):
    c = colourize(input_points)
    print(f"1b: {step:03d}/{steps}", end="\r")
    plt.figure(figsize=(6, 6))
    plt.scatter(
        space[step, :, 0],
        space[step, :, 1],
        edgecolor=c,
        facecolor="None",
        s=5,
        alpha=1,
        linewidth=1,
    )
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(mapping_animation_dir, f"{step:03d}.png"))
    plt.close()
print()
os.system(
    f"ffmpeg -r 20 -i {os.path.join(mapping_animation_dir, '%03d.png')}"
    f" -crf 15 {os.path.join(mapping_animation_dir, 'mapping.mp4')}"
)
