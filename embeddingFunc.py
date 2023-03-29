import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)

def load_image_paths_from_file(file_path):
    with open(file_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
    return image_paths


# Load the pre-trained Siamese model
siamese_model_path = 'C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\triplet_siam.h5'
siamese_model = load_model(siamese_model_path, custom_objects={'triplet_loss': triplet_loss})
image_embeddings_path = 'image_embeddings.npy'
image_embeddings = np.load(image_embeddings_path)
image_paths_file = 'image_paths.txt'
image_paths = load_image_paths_from_file(image_paths_file)

# Get the embedding model from the Siamese model
embedding_model = Model(inputs=siamese_model.get_layer('model_6').input, outputs=siamese_model.get_layer('model_6').get_layer('dense_4').output)




def embedding_function(image):
    image = np.expand_dims(image, axis=0)
    embedding_model.summary()  
    embedding = embedding_model.predict(image)
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
