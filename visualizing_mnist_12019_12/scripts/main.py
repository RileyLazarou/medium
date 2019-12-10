import os
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from PIL import Image
from matplotlib.cm import get_cmap

sns.set()

## Constants and parameters
ACTIVATION = 'relu'
CMAP = get_cmap('Greys_r')
EPOCHS = 4
BATCH_SIZE = 1024
PLOT_EVERY = 1
'''
ACTIVATION = 'tanh'
CMAP = get_cmap('seismic_r')
EPOCHS = 8
BATCH_SIZE = 1024
PLOT_EVERY = 2
'''
DATA_DIR = f'../{ACTIVATION}/data_log/'
MODELS_DIR = f'../{ACTIVATION}/models/'
TEMP_DIR = f'../{ACTIVATION}/temp/'
IMAGES_DIR = f'../{ACTIVATION}/visualizations/'

## Set up directories
for i in [DATA_DIR, TEMP_DIR, IMAGES_DIR, MODELS_DIR]:
    if not os.path.exists(i):
        os.makedirs(i)
for i in range(1,5):
    if not os.path.exists(f'{IMAGES_DIR}conv{i}/'):
        os.makedirs(f'{IMAGES_DIR}conv{i}/')

## Load and process data
# Load mnist from keras
(x_train, y_train_raw), (x_test, y_test_raw) = mnist.load_data()

# Reshape to (?, 28, 28, 1)
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Normalize images to [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# One-hot encode labels
y_train = np.zeros((len(y_train_raw), 10))
y_train[np.arange(y_train.shape[0]), y_train_raw] = 1
y_test = np.zeros((len(y_test_raw), 10))
y_test[np.arange(y_test.shape[0]), y_test_raw] = 1

# Curate example data for visualization
example_indices = [10, 5, 35, 30, 4, 15, 11, 17, 268, 12,]
examples = np.array([x_test[x] for x in example_indices])

def double_resolution(im):
    new_im = np.empty((im.shape[0]*2, im.shape[1]*2, *im.shape[2:]))
    new_im[0::2, 0::2] = im
    new_im[1::2, 0::2] = im
    new_im[0::2, 1::2] = im
    new_im[1::2, 1::2] = im
    return new_im

def plot_activations(activations):
    f, ax = plt.subplots(activations.shape[-1], activations.shape[0])
    extreme = np.max(np.abs(activations))
    for i in range(activations.shape[-1]):
        for j in range(activations.shape[0]):
            ax[i, j].imshow(activations[j, :, :, i], vmin=0, vmax=extreme, cmap='inferno')
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_images(base_ims, activations, filename, vmin=0, vmax=1, imsize=28, frame=None):
    columns = len(base_ims)
    rows = 1 + activations.shape[-1]
    width = imsize * columns
    height = imsize * rows
    full_im = np.zeros((height, width, 4))
    full_im[..., 3] = 1.0

    #activations = np.clip(activations, vmin, vmax)
    activations = (activations - vmin) / (vmax - vmin)
    activations = np.clip(activations, 0, 1)

    for index, i in enumerate(base_ims):
        full_im[:imsize, index*imsize:(index+1)*imsize, :3] = i.reshape((imsize, imsize, 1))

    for column in range(columns):
        for row in range(1, rows):
            activation = activations[column, ..., row-1]
            while activation.shape[0] < imsize:
                activation = double_resolution(activation)
            activation = CMAP(activation)
            full_im[row*imsize:(row+1)*imsize, column*imsize:(column+1)*imsize] = activation

    full_im = double_resolution(full_im)
    im = Image.fromarray(np.array(full_im*255, dtype=np.uint8))
    im.save(filename)


## Create model
def make_classifier():
    '''
    Create an MNIST classifier
    Return the classifier and a sub-model with output at each major layer
    '''
    input_layer = layers.Input((28, 28, 1))
    conv1 = layers.Conv2D(filters=6, kernel_size=3, padding='same', activation=ACTIVATION)(input_layer)
    conv2 = layers.Conv2D(filters=6, kernel_size=3, padding='same', activation=ACTIVATION)(conv1)
    mp1 = layers.MaxPooling2D(pool_size=2, padding='same')(conv2) # 14x14x16
    conv3 = layers.Conv2D(filters=12, kernel_size=3, padding='same', activation=ACTIVATION)(mp1)
    conv4 = layers.Conv2D(filters=12, kernel_size=3, padding='same', activation=ACTIVATION)(conv3)
    mp2 = layers.MaxPooling2D(pool_size=2, padding='same')(conv4) # 7x7x32
    flattened = layers.Flatten()(mp2)
    dense1 = layers.Dense(64, activation='relu')(flattened)
    dense2 = layers.Dense(32, activation='relu')(dense1)
    output_layer = layers.Dense(10, activation='softmax')(dense2)

    model_conv1 = Model(input_layer, conv1)
    model_conv2 = Model(input_layer, conv2)
    model_conv3 = Model(input_layer, conv3)
    model_conv4 = Model(input_layer, conv4)
    model_dense1 = Model(input_layer, dense1)
    model_dense2 = Model(input_layer, dense2)
    classifier = Model(input_layer, output_layer)
    classifier.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model_conv1, model_conv2, model_conv3, model_conv4, model_dense1, model_dense2, classifier

model_conv1, model_conv2, model_conv3, model_conv4, model_dense1, model_dense2, classifier = make_classifier()

## Set up lists for storing data
data_dict = {
    'batch_number': [], 
    'train_loss': [], 
    'train_accuracy': [], 
    'test_loss': [], 
    'test_accuracy': [], 
    'conv1_activations': [], 
    'conv2_activations': [], 
    'conv3_activations': [], 
    'conv4_activations': [], 
    'dense1_activations': [], 
    'dense2_activations': [], 
    'classifier_activations': [], 
    'classifier_accuracy': [], 
    'classifier_loss': [], 
}

data_dict['batch_number'].append(0)
loss, accuracy = classifier.evaluate(x_test[:1024], y_test[:1024], batch_size=BATCH_SIZE, verbose=0)
data_dict['train_loss'].append(loss)
data_dict['train_accuracy'].append(accuracy)
loss, accuracy = classifier.evaluate(x_train[:1024], y_train[:1024], batch_size=BATCH_SIZE, verbose=0)
data_dict['test_loss'].append(loss)
data_dict['test_accuracy'].append(accuracy)
data_dict['conv1_activations'].append(model_conv1.predict(examples))
data_dict['conv2_activations'].append(model_conv2.predict(examples))
data_dict['conv3_activations'].append(model_conv3.predict(examples))
data_dict['conv4_activations'].append(model_conv4.predict(examples))
data_dict['dense1_activations'].append(model_dense1.predict(examples))
data_dict['dense2_activations'].append(model_dense2.predict(examples))
data_dict['classifier_activations'].append(classifier.predict(examples))

## Train model
batches = x_train.shape[0] // BATCH_SIZE
indices = np.arange(x_train.shape[0])
for epoch in range(EPOCHS):
    np.random.shuffle(indices)
    for batch_num in range(batches):
        print(f'Epoch {epoch+1:02d}/{EPOCHS:02d}; '
              f'{batch_num*BATCH_SIZE:05d}/{x_train.shape[0]:05d}',
              end = '\r')
        batch_indices = indices[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE]
        classifier.train_on_batch(x_train[batch_indices], y_train[batch_indices])
        if (epoch * batches + batch_num+1) % PLOT_EVERY == 0:
            loss, accuracy = classifier.evaluate(x_test[:1024], y_test[:1024], batch_size=BATCH_SIZE, verbose=0)
            data_dict['batch_number'].append(epoch * batches + batch_num+1)
            data_dict['train_loss'].append(loss)
            data_dict['train_accuracy'].append(accuracy)
            loss, accuracy = classifier.evaluate(x_train[:1024], y_train[:1024], batch_size=BATCH_SIZE, verbose=0)
            data_dict['test_loss'].append(loss)
            data_dict['test_accuracy'].append(accuracy)
            data_dict['conv1_activations'].append(model_conv1.predict(examples))
            data_dict['conv2_activations'].append(model_conv2.predict(examples))
            data_dict['conv3_activations'].append(model_conv3.predict(examples))
            data_dict['conv4_activations'].append(model_conv4.predict(examples))
            data_dict['dense1_activations'].append(model_dense1.predict(examples))
            data_dict['dense2_activations'].append(model_dense2.predict(examples))
            data_dict['classifier_activations'].append(classifier.predict(examples))
        
        

    print(f'Epoch {epoch+1:02d}/{EPOCHS:02d}; '
          f'{batch_num*BATCH_SIZE:05d}/{x_train.shape[0]:05d}',
          end = '\n')

print(classifier.evaluate(x_test, y_test))

model_conv1.save(f'{MODELS_DIR}model_conv1.h5')
model_conv2.save(f'{MODELS_DIR}model_conv2.h5')
model_conv3.save(f'{MODELS_DIR}model_conv3.h5')
model_conv4.save(f'{MODELS_DIR}model_conv4.h5')
model_dense1.save(f'{MODELS_DIR}model_dense1.h5')
model_dense2.save(f'{MODELS_DIR}model_dense2.h5')
classifier.save(f'{MODELS_DIR}classifier.h5')

data_dict = {key: np.array(value) for key, value in data_dict.items()}

plt.figure()
plt.plot(data_dict['batch_number'], data_dict['train_accuracy'], label='train_accuracy', c='maroon')
plt.plot(data_dict['batch_number'], data_dict['test_accuracy'], label='test_accuracy', c='darkorange')
plt.plot(data_dict['batch_number'], data_dict['train_loss'], label='train_loss', c='darkgreen')
plt.plot(data_dict['batch_number'], data_dict['test_loss'], label='test_loss', c='darkturquoise')
plt.legend()
plt.xlabel('Training Step')
plt.ylabel('Metric')
plt.savefig(f'{IMAGES_DIR}metrics.png')
plt.close()

for conv in range(1,5):
    print(f'Saving images for conv{conv}')
    biggest = np.max(data_dict[f'conv{conv}_activations'])
    #biggets = 1
    smallest = np.min(data_dict[f'conv{conv}_activations'])
    smallest = 0
    for index, i in enumerate(data_dict[f'conv{conv}_activations']):
        plot_images(examples, i, f'{IMAGES_DIR}conv{conv}/act.conv{conv}.{index:05d}.png', vmin=smallest, vmax=biggest)
    os.system(f'ffmpeg -r 8 -i {IMAGES_DIR}conv{conv}/act.conv{conv}.%05d.png'
              ''' -vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}':'''
              ''' start_number=1: x=0: y=0: fontcolor=black: fontsize=20:'''
              ''' box=1: boxcolor=white: boxborderw=5"'''
              f' -crf 15 {IMAGES_DIR}conv{conv}.mp4')
'''
print('Begin JSON dump...')
with open(f'{DATA_DIR}data.json', 'w') as f:
    json.dump({key: value.tolist() for key, value in data_dict.items()}, f)
print('Done!')
'''


