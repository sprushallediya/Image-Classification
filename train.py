import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from model import build_cnn_model
from utils import get_data_augmenter, plot_training_history, print_evaluation_report
import numpy as np

# === Load CIFAR-10 Data ===
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# === Data Augmentation ===
augmenter = get_data_augmenter()
train_generator = augmenter.flow(x_train, y_train, batch_size=64)

# === Build Model ===
model = build_cnn_model()

# === Compile Model ===
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# === Train Model ===
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=(x_test, y_test)
)

# === Plot Accuracy/Loss ===
plot_training_history(history)

# === Evaluate on Test Set ===
y_pred_logits = model.predict(x_test)
y_pred = np.argmax(y_pred_logits, axis=1)

print_evaluation_report(y_test, y_pred, class_names)

# === Save Model ===
model.save("saved_model/cifar10_model.h5")
print("Model saved to 'saved_model/cifar10_model.h5'")


#Load CIFAR-10
#Preprocess and augment the data
#Build the model
#Train and validate
#Plot training performance
#Evaluate with confusion matrix + classification report
#Save the model 
