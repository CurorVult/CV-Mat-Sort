import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications.xception import Xception

def extract_class_label(file_name):
    return os.path.splitext(file_name)[0]

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

# Load your dataset
data_dir = 'C:\\Users\\Sean\\Documents\\GitHub\\CV-Mat-Sort\\Phyrexia_ All Will Be One_images'
image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split dataset into train and test sets
train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
unique_labels = set([extract_class_label(file_name) for file_name in train_files])
num_classes = len(unique_labels)
print("Number of Classes")
print(num_classes)

# Load the Xception base model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a custom model using the Xception base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Freeze the base model layers (use the pre-trained weights)
base_model.trainable = False

# ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# Update train_generator and validation_generator
batch_size = 32
train_generator = custom_generator(train_files, data_dir, train_datagen, (224, 224), batch_size, 'categorical')
validation_generator = custom_generator(test_files, data_dir, train_datagen, (224, 224), batch_size, 'categorical')
#Increase the number of additional Images generated 
steps_per_epoch = len(train_files) // batch_size * 3 

# Compile and train the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 50
model.fit(train_generator,
          epochs=epochs,
          validation_data=validation_generator)

# Save the custom CNN model
model.save('mtgembed.h5')
