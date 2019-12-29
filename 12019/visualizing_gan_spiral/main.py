import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

NUM_BATCHES = 60000
BATCH_SIZE = 512
PLOT_EVERY = 100
GRID_RESOLUTION = 400


def target_function(Z):
    '''
    Map Z ([-1,1], [-1,1]) to a spiral
    '''
    r = 0.05 + 0.90*(Z[:, 0]+1)/2 + Z[:, 1]*0.05
    theta = 3 * Z[:, 0] * np.pi
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
    '''
    Build a generator mapping (R, R) to ([-1,1], [-1,1])
    '''
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(512)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(2, activation='tanh')(X)
    G = Model(input_layer, output_layer)
    return G


def build_discriminator():
    '''
    Build a discriminator mapping (R, R) to [0, 1]
    '''
    input_layer = layers.Input((2,))
    X = input_layer
    for i in range(3):
        X = layers.Dense(512)(X)
        X = layers.LeakyReLU(0.1)(X)
    output_layer = layers.Dense(1, activation='sigmoid')(X)
    D = Model(input_layer, output_layer)
    D.compile(Adam(learning_rate=0.001, beta_1=0.5),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return D

def build_GAN(G, D):
    '''
    Given a generator and a discriminator, build a GAN
    '''
    D.trainable = False
    input_layer = layers.Input((2,))
    X = G(input_layer)
    output_layer = D(X)
    GAN = Model(input_layer, output_layer)
    GAN.compile(Adam(learning_rate=0.0002, beta_1=0.5),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
    return GAN

grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, 2))
grid[:, :, 0] = np.linspace(-1, 1, GRID_RESOLUTION).reshape((1, -1))
grid[:, :, 1] = np.linspace(1, -1, GRID_RESOLUTION).reshape((-1, 1))
flat_grid = grid.reshape((-1, 2))
test_samples = sample_from_target_function(5000)
test_noise = generate_noise(5000)
def plot(G, D, GAN, step, step_count, D_accuracy, D_loss, G_accuracy, G_loss, filename):
    '''
    Plots for the GAN gif
    '''
    f, ax = plt.subplots(2, 2, figsize=(8,8))
    f.suptitle(f'       {step:05d}', fontsize=10)


    # [0, 0]: plot loss and accuracy
    ax[0, 0].plot(step_count, 
                    G_loss,
                    label='G loss',
                    c='darkred',
                    zorder=50,
                    alpha=0.8,)
    ax[0, 0].plot(step_count, 
                    G_accuracy,
                    label='G accuracy',
                    c='lightcoral',
                    zorder=40,
                    alpha=0.8,)
    ax[0, 0].plot(step_count, 
                    D_loss,
                    label='D loss',
                    c='darkblue',
                    zorder=55,
                    alpha=0.8,)
    ax[0, 0].plot(step_count, 
                    D_accuracy,
                    label='D accuracy',
                    c='cornflowerblue',
                    zorder=45,
                    alpha=0.8,)
    ax[0, 0].set_xlim(-5, NUM_BATCHES+5)
    ax[0, 0].set_ylim(-0.1, 2.1)
    ax[0, 0].legend(loc=1)

    # [0, 1]: Plot actual samples and fake samples
    fake_samples = G.predict(test_noise, batch_size=len(test_noise))
    ax[0, 1].scatter(test_samples[:, 0], 
                     test_samples[:, 1], 
                     edgecolor='blue', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label='Real')
    ax[0, 1].scatter(fake_samples[:, 0], 
                     fake_samples[:, 1], 
                     edgecolor='red', facecolor='None', s=5, alpha=1, 
                     linewidth=1, label='GAN')
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
    plt.savefig(filename)
    plt.close()

## Set up directories
paths = ['ims', 
            'ims/1', 
                'ims/1/a', 'ims/1/b', 
            'ims/2', 
                'ims/2/a', 'ims/2/b', 'ims/2/c',
            'ims/3'
        ]
for i in paths:
    if not os.path.exists(i):
        os.makedirs(i)

## Part 1: Visualize the problem
### 1a: Input and output distributions
f, ax = plt.subplots(1, 2, figsize=(12,6))
input_points = generate_noise(5000)
output_points = target_function(input_points)
ax[0].scatter(input_points[:, 0], input_points[:, 1], edgecolor='k', facecolor='None', s=5, alpha=1, linewidth=1)
ax[0].set_xlim(-1, 1)
ax[0].set_ylim(-1, 1)
ax[0].set_aspect(1)
ax[1].scatter(output_points[:, 0], output_points[:, 1], edgecolor='blue', facecolor='None', s=5, alpha=1, linewidth=1)
ax[1].set_xlim(-1, 1)
ax[1].set_ylim(-1, 1)
ax[1].set_aspect(1)
plt.tight_layout()
plt.savefig('ims/1/a/1a.png')
plt.close()

### 1b: 1a, but as a gif
steps = 300
space = np.linspace(input_points, output_points, steps)
c_space = np.linspace(0, 255, steps)
for step in range(steps):
    c = f'#0000{int(c_space[step]):02x}'
    print(f'1b: {step:03d}/{steps}', end='\r')
    plt.figure(figsize=(6,6))
    plt.scatter(space[step, :, 0], space[step, :, 1], edgecolor=c, facecolor='None', s=5, alpha=1, linewidth=1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(f'ims/1/b/1b.{step:03d}.png')
    plt.close()
print()
os.system(f'ffmpeg -r 20 -i ims/1/b/1b.%03d.png'
              f' -crf 15 ims/1/b/1b.mp4')


## Part 2: Demonstrate capacity
### 2a: Generator capacity
G = build_generator()
G.compile(Adam(learning_rate=0.001), loss='mse')
steps = 600
for step in range(steps):
    print(f'2a: {step:03d}/{steps}', end='\r')
    noise = generate_noise(BATCH_SIZE)
    target = target_function(noise)
    G.train_on_batch(noise, target)

    generated_points = G.predict(input_points, batch_size=len(input_points))
    plt.figure(figsize=(6,6))
    plt.title(f'Step {step:03d}')
    plt.scatter(output_points[:, 0], output_points[:, 1], 
                edgecolor='blue', facecolor='None', s=5, alpha=1, 
                linewidth=1, label='Real')
    plt.scatter(generated_points[:, 0], generated_points[:, 1], 
                edgecolor='red', facecolor='None', s=5, alpha=1, 
                linewidth=1, label='Generated')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect(1)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(f'ims/2/a/2a.{step:03d}.png')
    plt.close()
print()
os.system(f'ffmpeg -r 20 -i ims/2/a/2a.%03d.png'
              f' -crf 15 ims/2/a/2a.mp4')

### 2b: Discriminator capacity
D = build_discriminator()
D.compile(Adam(learning_rate=0.001), loss='mse')
steps = 600
count = 0
for step in range(steps):
    print(f'2b: {step:03d}/{steps}', end='\r')
    noise = generate_noise(BATCH_SIZE // 2)
    real = target_function(noise)
    samples = np.concatenate((noise, real), axis=0)
    labels = np.concatenate((np.zeros((BATCH_SIZE//2, 1)), 
                            np.ones((BATCH_SIZE//2, 1))), 
                            axis=0)
    D.train_on_batch(samples, labels)
    if step % 2 == 0:
        confidences = D.predict(flat_grid, batch_size=len(flat_grid)).reshape((GRID_RESOLUTION, GRID_RESOLUTION))
        plt.figure(figsize=(6,6))
        plt.title(f'Step {step:03d}')
        plt.imshow(confidences, cmap='seismic_r', vmin=0, vmax=1)
        plt.xticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4), np.linspace(-1, 1, 5))
        plt.yticks(np.arange(0, GRID_RESOLUTION+1, GRID_RESOLUTION//4), np.linspace(1, -1, 5))
        plt.gca().set_aspect(1)
        plt.tight_layout()
        plt.savefig(f'ims/2/b/2b.{count:03d}.png')
        plt.close()
        count += 1
print()
os.system(f'ffmpeg -r 20 -i ims/2/b/2b.%03d.png'
              f' -crf 15 ims/2/b/2b.mp4')

### 2c: real vs fake overlay
plt.scatter(output_points[:, 0], output_points[:, 1], edgecolor='blue', 
            facecolor='None', s=5, alpha=1, 
            linewidth=1, label='real')
plt.scatter(input_points[:, 0], input_points[:, 1], edgecolor='red', 
            facecolor='None', s=5, alpha=1, 
            linewidth=1, label='fake')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('ims/2/c/2c1.png')
plt.close()

plt.scatter(output_points[:, 0], output_points[:, 1], edgecolor='blue', 
            facecolor='None', s=5, alpha=1, 
            linewidth=1, label='real')
plt.scatter(input_points[:, 0]*0.02+0.5, input_points[:, 1]*0.02+0.5, edgecolor='red', 
            facecolor='None', s=5, alpha=1, 
            linewidth=1, label='fake')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.gca().set_aspect(1)
plt.tight_layout()
plt.savefig('ims/2/c/2c2.png')
plt.close()

## Part 3: GAN
G = build_generator()
D = build_discriminator()
GAN = build_GAN(G, D)

step_count = []
D_accuracy = []
G_accuracy = []
D_loss = []
G_loss = []
count = 0
for step in range(NUM_BATCHES):
    # Train discriminator
    D.trainable = True
    real_data = sample_from_target_function(BATCH_SIZE // 2)
    fake_data = G.predict(generate_noise(BATCH_SIZE // 2), batch_size=BATCH_SIZE // 2)
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
        plot(G=G, 
             D=D,
             GAN=GAN,
             step=step,
             step_count=step_count,
             D_accuracy=D_accuracy,
             D_loss=D_loss,
             G_accuracy=G_accuracy,
             G_loss=G_loss,
             filename=f'ims/3/3.{count:03d}.png')
        count += 1

os.system(f'ffmpeg -r 20 -i ims/3/3.%03d.png'
              f' -crf 15 ims/3/3.mp4')
G.save
G.save("ims/3/G.h5")
D.save("ims/3/D.h5")