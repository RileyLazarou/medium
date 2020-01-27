import os
import colorsys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


NUM_BATCHES = 15000
BATCH_SIZE = 128
PLOT_EVERY = 30
LATENT_SPACE = 16
FILE = ".".join(os.path.basename(__file__).split(".")[:-1])

# Load and transform data
(x_train, y_train), (x_test, y_test) = load_data()
x_train = x_train.reshape((-1, 28, 28, 1))
x_train = x_train / 255
x_test = x_test.reshape((-1, 28, 28, 1))
x_test = x_test / 255


def plot_images(ims, num, filename, inverse=True):
    """Given a list of (equally-shaped) images
    Save them as `filename` in a `num`-by-`num` square
    """

    ims = np.array(ims)
    if len(ims) < num ** 2:
        indices = np.arange(len(ims))
    else:
        indices = np.arange(num ** 2)
    image_height = ims.shape[1]
    image_width = ims.shape[2]

    full_im = np.zeros((num * image_height, num * image_width, 4))
    for index, i in enumerate(indices):
        column = index % num
        row = index // num
        full_im[
            row * image_height : (row + 1) * image_height,
            column * image_width : (column + 1) * image_width,
        ] = ims[i, :, :]
    if inverse:
        full_im = 1 - full_im
    full_im[:, :, 3] = 1

    im = Image.fromarray(np.array(full_im * 255, dtype=np.uint8))
    im.save(filename)


def generate_noise(samples):
    """Random uniform(-1, 1) samples with LATENT_SPACE dimensions"""
    return np.random.uniform(-1, 1, (samples, LATENT_SPACE))


def sample_from_target_function(samples):
    """sample from the target function"""
    X = x_train[np.random.randint(0, x_train.shape[0], samples)]
    return X


def build_generator():
    input_layer = layers.Input(shape=(LATENT_SPACE,))

    X = layers.Dense(4 * 4 * 256)(input_layer)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Reshape((4, 4, 256))(X)
    X = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(X)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(X)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(X)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Conv2D(64, (5, 5), strides=(1, 1), padding="valid")(X)
    X = layers.LeakyReLU(alpha=0.2)(X)
    output_layer = layers.Conv2D(
        1, (7, 7), activation="sigmoid", padding="same"
    )(X)

    G = Model(input_layer, output_layer)
    G.compile(loss="binary_crossentropy", optimizer=Adam(lr=2e-4, beta_1=0.5))
    return G


def build_discriminator():
    input_layer = layers.Input(shape=(28, 28, 1))

    X = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(input_layer)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Dropout(0.4)(X)
    X = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")(X)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Dropout(0.4)(X)
    X = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(X)
    X = layers.LeakyReLU(alpha=0.2)(X)
    X = layers.Dropout(0.4)(X)
    X = layers.Flatten()(X)
    output_layer = layers.Dense(1, activation="sigmoid")(X)

    D = Model(input_layer, output_layer)
    D.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=2e-4, beta_1=0.5),
        metrics=["accuracy"],
    )
    return D


def build_GAN(G, D):
    D.trainable = False
    input_layer = layers.Input(shape=(LATENT_SPACE,))
    constructed = G(input_layer)
    prediction = D(constructed)
    model = Model(input_layer, prediction)
    model.compile(
        Adam(2e-4, beta_1=0.5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


## Set up directories
image_dir = os.path.join("ims")
script_dir = os.path.join(image_dir, FILE)
stills_dir = os.path.join(script_dir, "1_stills")
training_dir = os.path.join(script_dir, "2_training")
mapping_dir = os.path.join(script_dir, "3_mapping")

for i in [
    image_dir,
    script_dir,
    stills_dir,
    training_dir,
    mapping_dir,
]:
    if not os.path.exists(i):
        os.makedirs(i)

test_samples = sample_from_target_function(400)
test_noise = generate_noise(400)

## Part 1: Visualize the problem
### 1a: 10x10 random images from the training set
plot_images(
    test_samples, 10, os.path.join(stills_dir, "samples.png"), inverse=True
)

## Part 2: train GAN
if os.path.exists(os.path.join(training_dir, "G.h5")):
    G = load_model(os.path.join(training_dir, "G.h5"))
    D = load_model(os.path.join(training_dir, "D.h5"))
    GAN = build_GAN(G, D)
    skip_training = True
else:
    G = build_generator()
    D = build_discriminator()
    GAN = build_GAN(G, D)
    skip_training = False

count = 0
loss_g = 0
loss_d = 0
alpha = 0.8
for step in range(NUM_BATCHES):
    if skip_training:
        break
    if step % 10 == 0:
        print(
            f"step {step}/{NUM_BATCHES}; G={loss_g:.4f}, D={loss_d:.4f}",
            end="\r",
        )
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
    loss_d = loss_d * alpha + _D_loss * (1 - alpha)

    # Train generator
    D.trainable = False
    noise = generate_noise(BATCH_SIZE)
    labels = np.ones((BATCH_SIZE, 1))
    _G_loss, _G_accuracy = GAN.train_on_batch(noise, labels)
    loss_g = loss_g * alpha + _G_loss * (1 - alpha)

    if step % PLOT_EVERY == 0:
        samples = G.predict(test_noise)
        plot_images(
            samples,
            10,
            os.path.join(training_dir, f"training.{count:05d}.png"),
        )
        count += 1
print()

G.save(os.path.join(training_dir, "G.h5"))
D.save(os.path.join(training_dir, "D.h5"))

os.system(
    f"ffmpeg -r 20 -i {os.path.join(training_dir, 'training.%05d.png')}"
    f" -crf 15 {os.path.join(training_dir, 'training.mp4')}"
)

## Part 3: GAN mapping
for i in "abcde":
    this_mapping_dir = os.path.join(mapping_dir, i)
    if not os.path.exists(this_mapping_dir):
        os.makedirs(this_mapping_dir)
    original_vec = generate_noise(1)
    dims = np.arange(LATENT_SPACE)
    np.random.shuffle(dims)

    space = np.tile(original_vec.reshape(1, LATENT_SPACE), (600, 1))
    space[:, dims[0]] = -1
    space[:, dims[1]] = -1
    space[:150, dims[0]] = np.sin(np.linspace(-np.pi / 2, np.pi / 2, 150))
    space[150:300, dims[0]] = space[:150, dims[0]][::-1]
    space[300:450, dims[1]] = np.sin(np.linspace(-np.pi / 2, np.pi / 2, 150))
    space[450:, dims[1]] = space[300:450, dims[1]][::-1]
    ims = G.predict(space)
    for j in range(600):
        fig, ax = plt.subplots(2, figsize=(4, 4.5))
        ax[0].imshow(ims[j, ..., 0], cmap="Greys")
        ax[0].axis("off")
        for k in range(4):
            ax[1].plot([-1, 1], [-k, -k], c="k", zorder=50)
            ax[1].scatter(
                [-1, 1], [-k, -k], c="k", marker="o", s=30, zorder=50
            )
            ax[1].scatter(
                [space[j, dims[k]]],
                [-k],
                facecolor="b",
                edgecolor="k",
                marker="o",
                s=50,
                zorder=60,
            )
        ax[1].set_ylim(-3.2, 0.2)
        ax[1].set_xlim(-1.05, 1.05)
        # ax[1].set_aspect("equal")
        ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(this_mapping_dir, f"im.{j:03d}.png"),
            bbox_inches=0,
            pad_inches=0,
        )
        plt.close()
    os.system(
        f"ffmpeg -r 30 -i {os.path.join(this_mapping_dir, 'im.%03d.png')}"
        f" -crf 15 {os.path.join(this_mapping_dir, i + '.mp4')}"
    )
