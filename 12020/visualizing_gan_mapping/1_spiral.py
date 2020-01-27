import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import colorsys

NUM_BATCHES = 60000
BATCH_SIZE = 512
PLOT_EVERY = 100
GRID_RESOLUTION = 400
FILE = ".".join(os.path.basename(__file__).split(".")[:-1])


def target_function(Z):
    """
    Map Z ([-1,1], [-1,1]) to a spiral
    """
    r = 0.05 + 0.90 * (Z[:, 0] + 1) / 2 + Z[:, 1] * 0.05
    theta = 3 * Z[:, 0] * np.pi
    results = np.zeros((Z.shape[0], 2))
    results[:, 0] = r * np.cos(theta)
    results[:, 1] = r * np.sin(theta)
    return results


def generate_noise(samples):
    """
    Generate `samples` samples of uniform noise in
    ([-1,1], [-1,1])
    """
    return np.random.uniform(-1, 1, (samples, 2))


def sample_from_target_function(samples):
    """
    sample from the target function
    """
    Z = generate_noise(samples)
    return target_function(Z)


def build_generator():
    """
    Build a generator mapping (R, R) to ([-1,1], [-1,1])
    """
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(512)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(2, activation="tanh")(X)
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


## Set up directories
image_dir = os.path.join("ims")
script_dir = os.path.join("ims", FILE)
target_function_dir = os.path.join("ims", FILE, "1")
gan_training_dir = os.path.join("ims", FILE, "2")
gan_function_dir = os.path.join("ims", FILE, "3")

for i in [target_function_dir, gan_training_dir, gan_function_dir]:
    for j in "ab":
        dirname = os.path.join(i, j)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
grid[:, :, 1] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
flat_grid = grid.reshape((-1, 2))
test_samples = sample_from_target_function(5000)
test_noise = generate_noise(5000)


## Part 1: Visualize the problem
### 1a: Input and output distributions
f, ax = plt.subplots(1, 2, figsize=(12, 6))
input_points = generate_noise(10000)
output_points = target_function(input_points)
c = [
    colorsys.hsv_to_rgb(x * 0.9, 1, y * 0.8 + 0.2)
    for (x, y) in (input_points + 1) / 2
]
ax[0].scatter(
    input_points[:, 0],
    input_points[:, 1],
    edgecolor=c,
    facecolor="None",
    s=5,
    alpha=1,
    linewidth=1,
)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
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
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_aspect(1)
plt.tight_layout()
plt.savefig(os.path.join(target_function_dir, "a", "1a.png"))
plt.close()


### 1b: 1a, but as a gif
steps = 300
space = np.linspace(input_points, output_points, steps)
for step in range(steps):
    c = [
        colorsys.hsv_to_rgb(x * 0.9, 1, y * 0.8 + 0.2)
        for (x, y) in (input_points + 1) / 2
    ]
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
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(target_function_dir, "b", f"b.{step:03d}.png"))
    plt.close()
print()
im_filename = os.path.join(target_function_dir, "b", "b.%03d.png")
video_filename = os.path.join(target_function_dir, "b", "b.mp4")
os.system(f"ffmpeg -r 20 -i {im_filename}" f" -crf 15 {video_filename}")

## Part 2: GAN
G = build_generator()
D = build_discriminator()
GAN = build_GAN(G, D)

step_count = []
D_accuracy = []
G_accuracy = []
D_loss = []
G_loss = []
count = 0
test_noise = generate_noise(5000)
for step in range(NUM_BATCHES):
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
        samples = G.predict(test_noise)
        real = target_function(test_noise)
        c = [
            colorsys.hsv_to_rgb(x * 0.9, 1, y * 0.8 + 0.2)
            for (x, y) in (test_noise + 1) / 2
        ]

        plt.figure(figsize=(6, 6))
        plt.scatter(
            real[:, 0],
            real[:, 1],
            edgecolor="#BBBBBB",
            facecolor="None",
            s=5,
            alpha=1,
            linewidth=1,
        )
        plt.scatter(
            samples[:, 0],
            samples[:, 1],
            edgecolor=c,
            facecolor="None",
            s=5,
            alpha=1,
            linewidth=1,
        )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(gan_training_dir, "a", f"a.{count:05d}.png"))
        plt.close()
        count += 1


print()
os.system(
    f"ffmpeg -r 20 -i {os.path.join(gan_training_dir, 'a', 'a.%05d.png')}"
    f" -crf 15 {os.path.join(gan_training_dir, 'a', 'a.mp4')}"
)


## Part 1: Visualize the problem
### 1a: Input and output distributions
f, ax = plt.subplots(1, 2, figsize=(12, 6))
input_points = generate_noise(10000)
output_points = G.predict(input_points)
c = [
    colorsys.hsv_to_rgb(x * 0.9, 1, y * 0.8 + 0.2)
    for (x, y) in (input_points + 1) / 2
]
ax[0].scatter(
    input_points[:, 0],
    input_points[:, 1],
    edgecolor=c,
    facecolor="None",
    s=5,
    alpha=1,
    linewidth=1,
)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
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
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_aspect(1)
plt.tight_layout()
plt.savefig(os.path.join(gan_training_dir, "a", f"a.png"))
plt.close()


### 1b: 1a, but as a gif
steps = 300
space = np.linspace(input_points, output_points, steps)
for step in range(steps):
    c = [
        colorsys.hsv_to_rgb(x * 0.9, 1, y * 0.8 + 0.2)
        for (x, y) in (input_points + 1) / 2
    ]
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
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(gan_function_dir, "b", f"b.{step:03d}.png"))
    plt.close()
print()
im_filename = os.path.join(gan_function_dir, "b", "b.%03d.png")
video_filename = os.path.join(gan_function_dir, "b", "b.mp4")
os.system(f"ffmpeg -r 20 -i {im_filename}" f" -crf 15 {video_filename}")

G.save(f"{os.path.join(gan_function_dir, 'G.h5')}")
D.save(f"{os.path.join(gan_function_dir, 'D.h5')}")
