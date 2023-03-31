import os
import numpy as np
import tensorflow as tf
import random
from collections import defaultdict
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import MaxPooling2D, Dropout
import createTriplet
from createTriplet import augment_images, create_triplets
import cv2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import gc




# Import the necessary modules for pre-trained model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def triplet_loss(y_true, y_pred, alpha=0.2):
    embedding_size = y_pred.shape[-1] // 3
    anchor, positive, negative = y_pred[:, :embedding_size], y_pred[:, embedding_size:2*embedding_size], y_pred[:, 2*embedding_size:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)

def create_embedding_model(input_shape, l2_strength=0.01):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_strength))(x)
    return Model(inputs=base_model.input, outputs=x)

# Load images
path_to_directory = "/content/drive/MyDrive/Datasets/card_images"
image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Create a list of images and their corresponding class labels
images = []
labels = []
num_augmented_images = 6
for image_path in image_paths:
    augmented_images = augment_images(image_path, num_augmented_images=num_augmented_images)
    class_name = os.path.basename(image_path).split('_')[0]  # Extract class name from the image path
    images.extend(augmented_images)
    labels.extend([class_name] * len(augmented_images))

def load_image(image_data, target_size=(224, 224)):
    if isinstance(image_data, str):
        img = img_to_array(load_img(image_data, target_size=target_size))
    elif isinstance(image_data, np.ndarray):
        img = cv2.resize(image_data, target_size)
    else:
        raise TypeError(f"image_data should be a string or a numpy array, not {type(image_data)}")
    
    return preprocess_input(img)

train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=0.8, stratify=labels)


# Free up memory
del images
del labels
gc.collect()

# Create triplets for training and validation sets
train_anchor_images, train_positive_images, train_negative_images = create_triplets(train_images, train_labels)
val_anchor_images, val_positive_images, val_negative_images = create_triplets(val_images, val_labels)

# Free up memory
del train_images
del val_images
del train_labels
del val_labels
gc.collect()

loaded_train_anchor_images = np.array([load_image(img) for img in train_anchor_images])
loaded_train_positive_images = np.array([load_image(img) for img in train_positive_images])
loaded_train_negative_images = np.array([load_image(img) for img in train_negative_images])

# Load the images for the validation triplets
loaded_val_anchor_images = np.array([load_image(img) for img in val_anchor_images])
loaded_val_positive_images = np.array([load_image(img) for img in val_positive_images])
loaded_val_negative_images = np.array([load_image(img) for img in val_negative_images])

num_train_triplets = len(train_anchor_images)
num_val_triplets = len(val_anchor_images)

# Free up memory
del train_anchor_images
del train_positive_images
del train_negative_images
del val_anchor_images
del val_positive_images
del val_negative_images
gc.collect()

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

#Initial Learning Rate
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9

lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Compile and train the model
siamese_triplet_model.compile(optimizer=Adam(lr_schedule), loss=triplet_loss)

y_dummy_train = np.zeros(num_train_triplets)
y_dummy_val = np.zeros(num_val_triplets)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

siamese_triplet_model.fit(
    [loaded_train_anchor_images, loaded_train_positive_images, loaded_train_negative_images],
    y_dummy_train,
    validation_data=([loaded_val_anchor_images, loaded_val_positive_images, loaded_val_negative_images], y_dummy_val),
    batch_size=16,
    epochs=50,
    callbacks=[early_stopping]  # Pass the EarlyStopping callback to the fit method
)

siamese_triplet_model.save("triplet_siam.h5")

# Test the performance of the model
test_image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]
test_images = []
test_labels = []

for image_path in test_image_paths:
    test_image = load_image(image_path)
    test_images.append(test_image)
    test_labels.append(os.path.basename(image_path).split('_')[0])

test_images = np.array(test_images)

embeddings = embedding_model.predict(test_images)
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_classifier.fit(embeddings_normalized, test_labels)

# Evaluate the KNN classifier
accuracy = knn_classifier.score(embeddings_normalized, test_labels)
print("KNN classifier accuracy:", accuracy)