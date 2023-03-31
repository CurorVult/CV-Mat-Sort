from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2

def create_triplets(images, labels, num_triplets=1000):
    anchor_images, positive_images, negative_images = [], [], []

    # Create a dictionary to map card classes to their respective images.
    class_to_images = {}
    for img, label in zip(images, labels):
        if label not in class_to_images:
            class_to_images[label] = []
        class_to_images[label].append(img)

    # Filter out classes with less than 2 images
    valid_classes = [cls for cls in class_to_images.keys() if len(class_to_images[cls]) >= 2]
    if not valid_classes:
        raise ValueError("Not enough images per class for triplet generation. Please add more images or augment the dataset.")

    for _ in range(num_triplets):
        # Select a random class for the anchor and positive images
        anchor_class = random.choice(valid_classes)
        anchor_image, positive_image = random.sample(class_to_images[anchor_class], 2)
        anchor_images.append(anchor_image)
        positive_images.append(positive_image)

        # Select a random class for the negative image, making sure it's different from the anchor_class
        negative_class = random.choice([cls for cls in valid_classes if cls != anchor_class])
        negative_image = random.choice(class_to_images[negative_class])
        negative_images.append(negative_image)

    return anchor_images, positive_images, negative_images

def augment_images(image_path, num_augmented_images=1):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode='nearest'
    )

    # Load images into array and normalize
    image = img_to_array(load_img(image_path, target_size=(224, 224))) / 255.0
    image = image.reshape((1,) + image.shape)

    augmented_images = []
    for _ in range(num_augmented_images):
        for batch in datagen.flow(image, batch_size=1):
            aug_image = batch[0]
            augmented_images.append(aug_image)
            break

    return augmented_images


# Load images from Directory
path_to_directory = "/content/drive/MyDrive/Datasets/One_images"
image_paths = [os.path.join(path_to_directory, file) for file in os.listdir(path_to_directory) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Create a list of images and their corresponding class labels
images = []
labels = []
num_augmented_images = 4
for image_path in image_paths:
    augmented_images = augment_images(image_path, num_augmented_images=num_augmented_images)
    class_name = os.path.basename(image_path).split('_')[0]  # Extract class name from the image path
    images.extend(augmented_images)
    labels.extend([class_name] * len(augmented_images))

# Now you can use the create_triplets function on the images and labels
anchor_images, positive_images, negative_images = create_triplets(images, labels, num_triplets=1000)
