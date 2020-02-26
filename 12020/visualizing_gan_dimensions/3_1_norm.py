import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import os
import colorsys
import progressbar

LATENT_DIM = int(sys.argv[1])
RATE = int(sys.argv[2])
NUM_BATCHES = 600 * RATE
BATCH_SIZE = 512
PLOT_EVERY = 1 * RATE
FILE = ".".join(os.path.basename(__file__).split(".")[:-1])

THETA_MAPPING = np.arange(8).reshape((-1, 1)) / 8  # rotations


def generate_noise(samples):
    """
    Generate `samples` samples of uniform noise in
    ([0,1],)
    """
    noise = np.random.rand(samples, LATENT_DIM)
    return noise


def sample_from_target_function(samples):
    """
    sample from the target function
    """
    return np.random.normal(0, 1, (samples, 2))


def build_generator():
    """
    Build a generator mapping LATENT_DIM to ([-1,1], [-1,1])
    """
    input_layer = layers.Input((LATENT_DIM,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(256, activation='relu')(X)
    output_layer = layers.Dense(2)(X)
    G = Model(input_layer, output_layer)
    return G


def build_discriminator():
    """
    Build a discriminator mapping (R, R) to [0, 1]
    """
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(5):
        X = layers.Dense(128)(X)
        X = layers.LeakyReLU(0.01)(X)
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
    input_layer = layers.Input((LATENT_DIM,))
    X = G(input_layer)
    output_layer = D(X)
    GAN = Model(input_layer, output_layer)
    GAN.compile(
        Adam(learning_rate=0.0002, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return GAN


test_samples = sample_from_target_function(5000)
test_noise = generate_noise(5000)


image_dir = os.path.join("ims")
script_dir = os.path.join(image_dir, FILE)
RESULTS_DIR = os.path.join(script_dir, f'l{LATENT_DIM}_r{RATE}')


for i in [image_dir, script_dir, RESULTS_DIR,]:
    if not os.path.exists(i):
        os.makedirs(i)

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
    confidences = D.predict(fake_samples).flatten()
    c = 'r'
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
    plt.xlim(-4.2, 4.2)
    plt.ylim(-4.2, 4.2)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


## Part 1: GAN
G = build_generator()
G.summary()
D = build_discriminator()
D.summary()
GAN = build_GAN(G, D)

step_count = []
D_accuracy = []
G_accuracy = []
D_loss = []
G_loss = []
count = 0
for step in progressbar.progressbar(range(NUM_BATCHES)):
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
            filename=os.path.join(RESULTS_DIR, f"{count:06d}.png"),
        )
        count += 1
else:
    print()
    os.system(
        f"ffmpeg -r 20 -i {os.path.join(RESULTS_DIR,'%06d.png')}"
        f" -crf 15 {os.path.join(RESULTS_DIR, 'training.mp4')}"
    )
