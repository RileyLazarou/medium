import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

LATENT_DIMENSION = 32
IMAGE_DIMENSIONS = (28, 28, 1)
BATCH_SIZE = 512
EPOCHS = 75
LR_GAN = 2e-4
LR_AE = LR_GAN * 1e-1

# Load training and testing data
train_data = np.genfromtxt('resources/mnist-in-csv/mnist_train.csv', delimiter=',', skip_header=1)
y_train = train_data[:, 0].reshape((-1, 1))  # training labels as a columns vector
X_train = train_data[:, 1:].reshape((-1, *IMAGE_DIMENSIONS))
X_train = X_train / 255.0 # normalize to [0, 1]
X_train = 1 - X_train # inverting, for aesthetic reasons
del train_data

test_data = np.genfromtxt('resources/mnist-in-csv/mnist_test.csv', delimiter=',', skip_header=1)
y_test = test_data[:, 0].reshape((-1, 1))  # testing labels as a columns vector
X_test = test_data[:, 1:].reshape((-1, *IMAGE_DIMENSIONS))
X_test = X_test / 255.0 # normalize to [0, 1]
X_test = 1 - X_test # inverting, for aesthetic reasons
del test_data

def visualize_images(images, dim):
	height, width, depth = IMAGE_DIMENSIONS
	canvas = np.zeros((height * dim, width * dim, depth))
	for i in range(dim):
		for j in range(dim):
			canvas[i*height:(i+1)*height, j*width:(j+1)*width] = images[i*dim+j]
	if canvas.shape[2] == 1:
		canvas = canvas.reshape(canvas.shape[:2])
		plt.imshow(canvas, cmap='Greys_r')
	else:
		plt.imshow(canvas)
	plt.show()


def build_encoder():
	input_layer = layers.Input(shape=IMAGE_DIMENSIONS)

	_X = layers.Conv2D(8, kernel_size=3, strides=2)(input_layer)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2D(16, kernel_size=3, strides=2)(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2D(32, kernel_size=3, strides=2)(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Flatten()(_X)

	_X = layers.Dense(64)(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	output_layer = layers.Dense(LATENT_DIMENSION, activation='tanh')(_X)

	model = Model(input_layer, output_layer)
	model.compile('adam', loss='mse')
	return model

def build_generator():

	input_layer = layers.Input(shape=(LATENT_DIMENSION,))
	_X = layers.Dense(32)(input_layer)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Dense(4*4*32)(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Reshape((4,4,32))(_X)

	_X = layers.Conv2DTranspose(16, kernel_size=4, strides=2, padding='same')(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same')(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2DTranspose(8, kernel_size=4, strides=2, padding='same')(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2D(1, kernel_size=5, strides=1, padding='valid')(_X)
	output_layer = layers.Activation('sigmoid')(_X)

	model = Model(input_layer, output_layer)
	model.compile('adam', loss='mse')
	return model

def build_discriminator():
	input_layer = layers.Input(shape=IMAGE_DIMENSIONS)

	_X = layers.Conv2D(8, kernel_size=3, strides=2)(input_layer)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2D(16, kernel_size=3, strides=2)(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Conv2D(32, kernel_size=3, strides=2)(_X)
	_X = layers.BatchNormalization()(_X)
	_X = layers.LeakyReLU(0.1)(_X)
	_X = layers.Flatten()(_X)

	output_layer = layers.Dense(1, activation='sigmoid')(_X)

	model = Model(input_layer, output_layer)
	model.compile(Adam(LR_AE, beta_1=0.5), loss='binary_crossentropy')
	return model

def build_AE(E, G):
	input_layer = layers.Input(shape=IMAGE_DIMENSIONS)
	encoding = E(input_layer)
	reconstruction = G(encoding)
	model = Model(input_layer, reconstruction)
	model.compile(Adam(LR_AE), loss='mse')
	return model

def build_GAN(G, D):
	D.trainable = False
	input_layer = layers.Input(shape=(LATENT_DIMENSION,))
	constructed = G(input_layer)
	prediction = D(constructed)
	model = Model(input_layer, prediction)
	model.compile(Adam(LR_GAN, beta_1=0.5), loss='binary_crossentropy')
	return model


E = build_encoder()
print('Encoder')
E.summary()

G = build_generator()
print('Generator')
G.summary()

D = build_discriminator()
print('Discriminator')
D.summary()

AE = build_AE(E, G)
print('AE')
AE.summary()

GAN = build_GAN(G, D)
print('GAN')
GAN.summary()

# main loop
generate_noise = lambda : np.random.random((BATCH_SIZE, LATENT_DIMENSION)) * 2 - 1

batches = X_train.shape[0] // BATCH_SIZE
for epoch in range(EPOCHS):
	running_loss = 0
	for batch in range(batches):
		real_samples = X_train[np.random.randint(0, X_train.shape[0], BATCH_SIZE)]
		_loss = AE.train_on_batch(real_samples, real_samples)
		running_loss += _loss
		print(f'Epoch {epoch+1}: {batch*BATCH_SIZE:05d}/{X_train.shape[0]:05d}; loss={running_loss / (batch+1):.3f}', end='\r')
	test_reconstructions = AE.predict(X_test[:100])
	images = [x for pair in zip(X_test[:100], test_reconstructions) for x in pair]
visualize_images(images, 10)
















