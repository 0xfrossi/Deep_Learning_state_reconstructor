import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

from utils_functions import *
from plan import *

def preprocess(array):
    # Normalizza l'array in input e lo riforma nel formato appropriato

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

def noise(array):
    # Aggiunge rumore random a ciascuna immagine dell'array

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=array.shape)
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    # Mostra 10 immagini random per ciascun vettore

    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


# Preparazione del dataset

(train_data, _), (test_data, _) = mnist.load_data()

# Normalizzazione e riformazione del dataset

train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Copia rumorosa del dataset

noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Si mostra il dataset

display(train_data, noisy_train_data)

# Costruzione dell'autoencoder

input = layers.Input(shape=(28, 28, 1))

#Encoder
x = layers.Dense(32, activation='relu')(input)

#Decoder
x = layers.Dense(1, activation='sigmoid')(x)

#Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

# Encoding con dati puliti

# autoencoder.fit(
#     x=train_data,
#     y=train_data,
#     epochs=50,
#     batch_size=128,
#     shuffle=True,
#     validation_data=(test_data, test_data),
# )
#
# predictions = autoencoder.predict(test_data)
# display(test_data, predictions)

autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)

predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions)