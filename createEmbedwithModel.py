import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Load the saved model
model = load_model('mtgembed.h5')

# Remove the last Dense layer to get the embeddings
embedding_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def create_embeddings(directory_path):
    # Load images
    image_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

    # Calculate and save embeddings
    image_embeddings = [embedding_model.predict(preprocess_image(image_path)) for image_path in image_paths]
    image_embeddings = np.squeeze(np.array(image_embeddings))
    np.save('image_embeddings.npy', image_embeddings)

    # Save image paths
    with open('image_paths.txt', 'w') as f:
        for image_path in image_paths:
            f.write(image_path + '\n')

create_embeddings('C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images')
