import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import createTriplet

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)

def create_embedding_model(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation="relu")(input)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    return Model(inputs=input, outputs=x)
# Load images
path_to_directory = "C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images"
image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Create triplets
anchor_images, positive_images, negative_images = createTriplet.create_triplets(image_paths)

# Load and preprocess images
def load_image(path, target_size=(224, 224)):
    return img_to_array(load_img(path, target_size=target_size)) / 255.0

# Load the images for the triplets
anchor_images = np.array([load_image(path) for path in anchor_images])
positive_images = np.array([load_image(path) for path in positive_images])
negative_images = np.array([load_image(path) for path in negative_images])

# Create the base network for embeddings
input_shape = (224, 224, 3)
embedding_model = create_embedding_model(input_shape)

# Create the Siamese network with triplet loss
anchor_input = Input(shape=input_shape, name="anchor_input")
positive_input = Input(shape=input_shape, name="positive_input")
negative_input = Input(shape=input_shape, name="negative_input")

anchor_embedding = embedding_model(anchor_input)
positive_embedding = embedding_model(positive_input)
negative_embedding = embedding_model(negative_input)

outputs = tf.keras.layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)
siamese_triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=outputs)

# Compile and train the model
siamese_triplet_model.compile(optimizer=Adam(learning_rate=0.001), loss=triplet_loss)

y_dummy = np.zeros(len(anchor_images))
siamese_triplet_model.fit([anchor_images, positive_images, negative_images], y_dummy, batch_size=32, epochs=10)

siamese_triplet_model.save("triplet_siam.h5")