from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import os
import matplotlib.pyplot as plt

def augment_images(image_path, num_augmented_images=1):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='nearest'
    )

    # Load images into array and normalize
    image = img_to_array(load_img(image_path, target_size=(224, 224)))
    image = image.reshape((1,) + image.shape)

    augmented_images = []
    for i in range(num_augmented_images):
        for batch in datagen.flow(image, batch_size=1):
            aug_image = batch[0]
            augmented_images.append(aug_image)
            break

    # Visualize augmented images
    for i, aug_image in enumerate(augmented_images):
        plt.imshow(aug_image)
        plt.title(f'Augmented Image {i+1}')
        plt.show()

        # Save the image
        save_path = os.path.join(os.path.dirname(image_path), f'aug_{i}.png')
        save_img(save_path, aug_image)

image_path = r"C:\\Users\\Sean\\Downloads\\seedimage\\seedimage.png" # Provide your image path here
augment_images(image_path, num_augmented_images=5)