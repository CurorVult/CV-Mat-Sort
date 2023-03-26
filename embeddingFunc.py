import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

# Load images
path_to_directory = "/content/drive/MyDrive/Datasets/One_images"
image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Create the embedding function
def create_embedding_function(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu")(input)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    return Model(inputs=input, outputs=x)

input_shape = (162, 227, 3)
embedding_function = create_embedding_function(input_shape)

# Load and preprocess images
def load_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [162, 227])
    img = img / 255.0
    return img.numpy()

