import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load images 
path_to_directory = "path_to_directory"
image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

#Set the number of classes to the number of image files
num_classes = len(image_paths)
#Create two lists, one for our image pairs and one for the labels. Each pair is assigned 1 if matched and 0 otherwise
#Pairs are the image paired with itself and with each other image.
image_pairs, pair_labels = [], []
for i in range(len(image_paths)):
    for j in range(len(image_paths)):
        image_pairs.append((image_paths[i], image_paths[j]))
        pair_labels.append(1 if i == j else 0)

# Load and preprocess images
#Load each image pair into a numpy array of it's pixels
def load_image_pair(pair):
    img1 = img_to_array(load_img(pair[0], target_size=(325, 454))) / 255.0
    img2 = img_to_array(load_img(pair[1], target_size=(325, 454))) / 255.0
    return img1, img2

image_pairs = [load_image_pair(pair) for pair in image_pairs]

# Create train and test sets
split = int(0.8 * len(image_pairs))
train_pairs, train_labels = image_pairs[:split], pair_labels[:split]
test_pairs, test_labels = image_pairs[split:], pair_labels[split:]

# Create the Siamese network
def create_siamese_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation="relu")(input)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    return Model(inputs=input, outputs=x)

input_shape = (325, 454, 3)
base_network = create_siamese_network(input_shape)

#Train 2 different networks, one for each of our inputs
input1 = Input(shape=input_shape)
input2 = Input(shape=input_shape)

feature1 = base_network(input1)
feature2 = base_network(input2)

#Find the absolute difference between the features determined by our pair of networks
diff = tf.abs(feature1 - feature2)
#Use sigmoid activation to determine if the features match
output = Dense(1, activation="sigmoid")(diff)

model = Model(inputs=[input1, input2], outputs=output)

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

train_X1, train_X2 = zip(*train_pairs)
train_X1, train_X2 = np.array(train_X1), np.array(train_X2)
train_y = np.array(train_labels)

test_X1, test_X2 = zip(*test_pairs)
test_X1, test_X2 = np.array(test_X1), np.array(test_X2)
test_y = np.array(test_labels)

model.fit([train_X1, train_X2], train_y, validation_data=([test_X1, test_X2], test_y), batch_size=32, epochs=10)
