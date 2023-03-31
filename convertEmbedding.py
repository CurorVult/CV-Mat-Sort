from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from tensorflow.keras.applications.efficientnet import EfficientNetB0

class CustomEfficientNetB0(EfficientNetB0):
    pass

def create_embedding_model():
    input_shape = (224, 224, 3)
    base_model = CustomEfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(128, activation=None)(x)

    embedding_model = Model(inputs=base_model.input, outputs=output)
    embedding_model.compile(optimizer=Adam(learning_rate=0.001), loss=tfa.losses.TripletSemiHardLoss())

    return embedding_model

# Create a new model with the same architecture
new_embedding_model = create_embedding_model()

# Load the weights
new_embedding_model.load_weights("embedding_model_weights.h5")

# Save the model's architecture and weights as an HDF5 file
new_embedding_model.save('new_embedding_model.h5')