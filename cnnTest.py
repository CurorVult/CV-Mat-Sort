import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.layers import BatchNormalization

def extract_class_label(file_name):
    return os.path.splitext(file_name)[0]

# Load your dataset
data_dir = 'C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images'
image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split dataset into train and test sets
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
unique_labels = set([extract_class_label(file_name) for file_name in train_files])
num_classes = len(unique_labels)
print("Number of Classes")
print(num_classes)

# Create the custom CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

def custom_generator(file_list, data_dir, datagen, target_size, batch_size, class_mode):
    labels = [extract_class_label(file_name) for file_name in file_list]
    generator = datagen.flow_from_dataframe(
        pd.DataFrame({'filename': file_list, 'class_label': labels}),
        directory=data_dir,
        x_col='filename',
        y_col='class_label',
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        classes=list(unique_labels)
    )
    return generator

# Update train_generator and validation_generator
batch_size = 32
train_generator = custom_generator(train_files, data_dir, train_datagen, (224, 224), batch_size, 'categorical')
validation_generator = custom_generator(test_files, data_dir, train_datagen, (224, 224), batch_size, 'categorical')


# Compile and train the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 30
model.fit(train_generator,
          epochs=epochs,
          validation_data=validation_generator)

# Save the custom CNN model
model.save('mtgembed.h5')