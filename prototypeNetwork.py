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

image_embeddings = [embedding_function(np.expand_dims(load_preprocess_image(image_path), axis=0)) for image_path in image_paths]
image_embeddings = np.squeeze(np.array(image_embeddings))

# Given a query image, find the closest image in the dataset
def identify_image(query_image_path, image_embeddings, threshold=0.5):
    query_image = load_preprocess_image(query_image_path)
    query_embedding = embedding_function(np.expand_dims(query_image, axis=0))

    distances = np.linalg.norm(image_embeddings - query_embedding, axis=1)
    closest_index = np.argmin(distances)

    if distances[closest_index] < threshold:
        return image_paths[closest_index], distances[closest_index]
    else:
        return None, None

# Example usage
query_image_path = "/path/to/query/image.jpg"
identified_image, distance = identify_image(query_image_path, image_embeddings)
print(f"Identified image: {identified_image}, distance: {distance}")