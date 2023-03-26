import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
import embeddingFunc

# Load embeddings and image paths
loaded_image_embeddings = np.load('image_embeddings.npy')
with open('image_paths.txt', 'r') as f:
    loaded_image_paths = [line.strip() for line in f.readlines()]

# Given a query image, find the closest image in the dataset
def identify_image(query_image_path, image_embeddings, threshold=0.5):
    query_image = embeddingFunc.load_preprocess_image(query_image_path)
    query_embedding = embeddingFunc.embedding_function(np.expand_dims(query_image, axis=0))

    distances = np.linalg.norm(image_embeddings - query_embedding, axis=1)
    closest_index = np.argmin(distances)

    if distances[closest_index] < threshold:
        return loaded_image_paths[closest_index], distances[closest_index]
    else:
        return None, None

# Example usage
query_image_path = "/path/to/query/image.jpg"
identified_image, distance = identify_image(query_image_path, loaded_image_embeddings)
print(f"Identified image: {identified_image}, distance: {distance}")
