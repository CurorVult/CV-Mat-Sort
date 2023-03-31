#Note this code is meant to run in a Jupyter Notebook on Google Colab, be sure to run pip install on the following

#pip install optuna
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import gc
import kerastuner as kt
import cv2
import createTriplet
from createTriplet import augment_images, create_triplets
import optuna


# Import the necessary modules for pre-trained model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Triplet loss function
def triplet_loss(y_true, y_pred, alpha=0.2):
    embedding_size = y_pred.shape[-1] // 3
    anchor, positive, negative = y_pred[:, :embedding_size], y_pred[:, embedding_size:2*embedding_size], y_pred[:, 2*embedding_size:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)

# Create embedding model with EfficientNetB0
def create_embedding_model(input_shape, l2_strength=0.01):
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2_strength))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_strength))(x)
    return Model(inputs=base_model.input, outputs=x)

# Load images
path_to_directory = "/content/drive/MyDrive/Datasets/One_images"
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

# Load and preprocess image
def load_image(image_data, target_size=(224, 224)):
    if isinstance(image_data, str):
        img = img_to_array(load_img(image_data, target_size=target_size))
    elif isinstance(image_data, np.ndarray):
        img = cv2.resize(image_data, target_size)
    else:
        raise TypeError(f"image_data should be a string or a numpy array, not {type(image_data)}")
    
    return preprocess_input(img)

# Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=0.8, stratify=labels)

# Free up memory
del images
del labels
gc.collect()

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

def model_builder(trial):
    # Create the base network for embeddings
    input_shape = (224, 224, 3)

    # Hyperparameters
    l2_strength = trial.suggest_loguniform("l2_strength", 1e-6, 1e-2)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.5)
    dense_units = trial.suggest_categorical("dense_units", [128, 256, 512])

    embedding_model = create_embedding_model(input_shape, l2_strength=l2_strength)

    # Add dropout layers to the embedding model
    x = embedding_model.output
    for i in range(trial.suggest_int("num_dropout_layers", 1, 3)):
        x = Dropout(rate=dropout_rate, name=f"dropout_layer_{i+1}")(x)
    x = Dense(dense_units, activation="relu", kernel_regularizer=regularizers.l2(l2_strength), name="dense_layer")(x)
    modified_embedding_model = Model(inputs=embedding_model.input, outputs=x)

    # Create the Siamese network with triplet loss
    anchor_input = Input(shape=input_shape, name="anchor_input")
    positive_input = Input(shape=input_shape, name="positive_input")
    negative_input = Input(shape=input_shape, name="negative_input")

    anchor_embedding = modified_embedding_model(anchor_input)
    positive_embedding = modified_embedding_model(positive_input)
    negative_embedding = modified_embedding_model(negative_input)

    outputs = tf.keras.layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)
    siamese_triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=outputs)

    # Learning Rate
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])

    # Compile the model
    siamese_triplet_model.compile(optimizer=Adam(learning_rate), loss=triplet_loss)

    return siamese_triplet_model

def objective(trial):
    model = model_builder(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        [loaded_train_anchor_images, loaded_train_positive_images, loaded_train_negative_images],
        np.zeros(num_train_triplets),
        validation_data=([loaded_val_anchor_images, loaded_val_positive_images, loaded_val_negative_images], np.zeros(num_val_triplets)),
        batch_size=32,
        epochs=20,
        callbacks=[early_stopping]
    )

    loss = model.evaluate([loaded_val_anchor_images, loaded_val_positive_images, loaded_val_negative_images], np.zeros(num_val_triplets), verbose=0)
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

# Get the best hyperparameters
best_hyperparameters = study.best_trial.params
print("Best hyperparameters: ", best_hyperparameters)

# Build the best model with the best_hyperparameters and train it
best_model = model_builder(optuna.fixed_trial.FixedTrial(best_hyperparameters))


# Train the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

best_model.fit(
    [loaded_train_anchor_images, loaded_train_positive_images, loaded_train_negative_images],
    np.zeros(num_train_triplets),
    validation_data=([loaded_val_anchor_images, loaded_val_positive_images, loaded_val_negative_images], np.zeros(num_val_triplets)),
    batch_size=32,
    epochs=20,
    callbacks=[early_stopping]  # Pass the EarlyStopping callback to the fit method
)

# Get the best embedding model from the best Siamese network
best_embedding_model = Model(inputs=best_model.get_layer(index=2).get_input_at(0), outputs=best_model.get_layer(index=2).get_output_at(0))

# Save the best embedding model
best_model.save("best_siamese_model.h5")