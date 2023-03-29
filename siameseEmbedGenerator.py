import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_image(path, target_size=(224, 224)):
    return img_to_array(load_img(path, target_size=target_size)) / 255.0

def generate_embeddings(image_folder, embedding_model):
    image_paths = []
    embeddings = []

    for file in os.listdir(image_folder):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, file)
            image_paths.append(img_path)

            img = load_image(img_path)
            img_embedding = embedding_model.predict(np.expand_dims(img, axis=0))
            embeddings.append(img_embedding[0])

    return np.array(embeddings), image_paths

# Load the trained Siamese model
siamese_triplet_model = tf.keras.models.load_model("triplet_siam.h5", compile=False)

# Extract the embedding model from the Siamese model
embedding_model = siamese_triplet_model.layers[3]

# Set the path of your dataset folder
image_folder = 'C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images'

# Generate embeddings for your dataset
embeddings, image_paths = generate_embeddings(image_folder, embedding_model)

np.save('image_embeddings.npy', embeddings)

with open('image_paths.txt', 'w') as f:
    for path in image_paths:
        f.write(path + '\n')