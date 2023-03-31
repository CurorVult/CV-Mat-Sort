import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json

def load_image_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
    return image_paths

# Load the new_embedding_model
# Load the JSON architecture file
with open('new_embedding_model_architecture.json', 'r') as json_file:
    new_embedding_model_json = json_file.read()

# Create the model from the JSON string
new_embedding_model = model_from_json(new_embedding_model_json)

# Load the weights from the .h5 file
new_embedding_model.load_weights('new_embedding_model_weights.h5')

image_embeddings_path = 'image_embeddings.npy'
image_embeddings = np.load(image_embeddings_path)
image_paths_file = 'image_paths.txt'
image_paths = load_image_paths_from_file(image_paths_file)

# Use new_embedding_model for the embedding_function
def embedding_function(image):
    image = np.expand_dims(image, axis=0)
    new_embedding_model.summary()  
    embedding = new_embedding_model.predict(image)
    return embedding[0]

def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])  
    img = img / 255.0
    return img.numpy()

def load_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])  
    img = img / 255.0
    return img.numpy()

# Load images
# path_to_directory = "C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images"
# image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]
