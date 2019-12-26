import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

NUM_BATCHES = 5000
BATCH_SIZE = 512
PLOT_EVERY = 5
GRID_RESOLUTION = 400

def target_function(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a spiral
    '''
    r = 0.05 + 0.90*(Z[:, 0]+1)/2 + Z[:, 1]*0.05
    theta = 4 * Z[:, 0] * np.pi
    results = np.zeros((Z.shape[0], 2))
    results[:, 0] = r * np.cos(theta)
    results[:, 1] = r * np.sin(theta)
    return results

def generate_noise(samples):
    '''
    Generate `samples` samples of uniform noise in 
    ([-1,1], [-1,1])
    '''
    return np.random.uniform(-1, 1, (samples, 2))

def sample_from_target_function(samples):
    '''
    sample from the target function
    '''
    Z = generate_noise(samples)
    return target_function(Z)


def build_generator():
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(512)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(2, activation='tanh')(X)
    G = Model(input_layer, output_layer)
    G.compile(Adam(learning_rate=0.001),
                        loss='mse')
    return G


def build_discriminator():
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(512)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(1, activation='sigmoid')(X)
    D = Model(input_layer, output_layer)
    D.compile(Adam(learning_rate=0.0002),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return D

def build_GAN(G, D):
    D.trainable = False
    input_layer = layers.Input((2,))
    X = G(input_layer)
    output_layer = D(X)
    GAN = Model(input_layer, output_layer)
    GAN.compile(Adam(learning_rate=0.001, beta_1=0.5),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return GAN

grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
grid[:, :, 1] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
flat_grid = grid.reshape((-1, 2))
test_samples = sample_from_target_function(5000)
test_noise = generate_noise(5000)
def plot(G, D, GAN, step, D_accuracy, D_loss, G_accuracy, G_loss):
    
    f, ax = plt.subplots(2, 2, figsize=(8,8))
    f.suptitle(f'       {step:05d}', fontsize=10)


    # [0, 0]: plot loss and accuracy
    ax[0, 0].plot(np.arange(len(G_loss)), 
                    G_loss,
                    label='G loss',
                    c='darkred',
                    zorder=50,
                    alpha=0.8,)
    ax[0, 0].plot(np.arange(len(G_accuracy)), 
                    G_accuracy,
                    label='G accuracy',
                    c='lightcoral',
                    zorder=40,
                    alpha=0.8,)
    ax[0, 0].plot(np.arange(len(D_loss)), 
                    D_loss,
                    label='D loss',
                    c='darkblue',
                    zorder=55,
                    alpha=0.8,)
    ax[0, 0].plot(np.arange(len(D_accuracy)), 
                    D_accuracy,
                    label='D accuracy',
                    c='cornflowerblue',
                    zorder=45,
                    alpha=0.8,)
    ax[0, 0].set_xlim(-5, NUM_BATCHES+5)
    ax[0, 0].set_ylim(-0.1, 2.1)
    ax[0, 0].legend(loc=1)

    # [0, 1]: Plot actual samples and fake samples
    fake_samples = G.predict(test_noise)
    ax[0, 1].scatter(test_samples[:, 0], 
                     test_samples[:, 1], 
                     c='blue', 
                     label='real',
                     s=5)
    ax[0, 1].scatter(fake_samples[:, 0], 
                     fake_samples[:, 1], 
                     c='red', 
                     label='GAN',
                     s=2)
    ax[0, 1].set_xlim(-1, 1)
    ax[0, 1].set_ylim(-1, 1)
    ax[0, 1].legend(loc=1)
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    # [1, 0]: Confident real heatmap input
    confidences = GAN.predict(flat_grid, batch_size=BATCH_SIZE)
    confidences = confidences.reshape((GRID_RESOLUTION, GRID_RESOLUTION))
    ax[1, 0].imshow(confidences, vmin=0, vmax=1, cmap='seismic_r')
    ax[1, 0].set_xticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4))
    ax[1, 0].set_xticklabels(np.linspace(-1, 1, 5))
    ax[1, 0].set_yticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4))
    ax[1, 0].set_yticklabels(np.linspace(1, -1, 5))

    # [1, 1]: Confident real heatmap output
    confidences = D.predict(flat_grid, batch_size=BATCH_SIZE)
    confidences = confidences.reshape((GRID_RESOLUTION, GRID_RESOLUTION))
    ax[1, 1].imshow(confidences, vmin=0, vmax=1, cmap='seismic_r')
    ax[1, 1].set_xticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4))
    ax[1, 1].set_xticklabels(np.linspace(-1, 1, 5))
    ax[1, 1].set_yticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4))
    ax[1, 1].set_yticklabels(np.linspace(1, -1, 5))
    #ax[1, 1].set_xticks([0 ,10, 20], ['a', 'b', 'c'])

    plt.tight_layout()
    plt.savefig(f'ims/g{step:05d}.png')
    plt.close()

if not os.path.exists('ims'):
    os.mkdir('ims')

G = build_generator()
D = build_discriminator()
GAN = build_GAN(G, D)

D_accuracy = []
G_accuracy = []
D_loss = []
G_loss = []
for step in range(NUM_BATCHES):
    # Train discriminator
    D.trainable = True
    real_data = sample_from_target_function(BATCH_SIZE // 2)
    fake_data = G.predict(generate_noise(BATCH_SIZE // 2))
    data = np.concatenate((real_data, fake_data), axis=0)
    real_labels = np.ones((BATCH_SIZE // 2, 1))
    fake_labels = np.zeros((BATCH_SIZE // 2, 1))
    labels = np.concatenate((real_labels, fake_labels), axis=0)
    _D_loss, _D_accuracy = D.train_on_batch(data, labels)
    D_loss.append(_D_loss)
    D_accuracy.append(_D_accuracy)

    # Train generator
    D.trainable = False
    noise = generate_noise(BATCH_SIZE)
    labels = np.ones((BATCH_SIZE, 1))
    _G_loss, _G_accuracy = GAN.train_on_batch(noise, labels)
    G_loss.append(_G_loss)
    G_accuracy.append(_G_accuracy)

    if step % PLOT_EVERY == 0:
        plot(G=G, 
             D=D,
             GAN=GAN,
             step=step,
             D_accuracy=D_accuracy,
             D_loss=D_loss,
             G_accuracy=G_accuracy,
             G_loss=G_loss)
assert False
G_test = build_generator()
for i in range(5000):
    X = generate_noise(1024)
    Y = target_function(X)
    loss = G_test.train_on_batch(X, Y)
    print(f'{i:04d}: loss={loss:.4f}', end='\r')


X = generate_noise(5000)
Y = target_function(X)
Y_hat = G_test.predict(X)
plt.scatter(Y[:, 0], Y[:, 1], c='b', s=2)
plt.scatter(Y_hat[:, 0], Y_hat[:, 1], c='r', s=2)
plt.show()



