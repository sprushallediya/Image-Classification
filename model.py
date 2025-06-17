import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))  # No softmax, logits output

    return model


#3 convolutional blocks with ReLU + MaxPooling

#Flatten + Dense layers at the end

#Output layer returns logits (we'll apply softmax during evaluation)
