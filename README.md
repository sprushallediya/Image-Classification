# 🔍 CIFAR-10 Image Classifier Web App

This project is a simple and stylish **image classification web app** built using **TensorFlow, Streamlit, and Python**. It uses a Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset to recognize 10 types of objects from uploaded images.

---

##  Demo

> Upload an image (32x32 pixels or larger) and the app will predict one of these classes:
> 
> `['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']`

---

##  Features

-  Real-time image classification using a trained CNN
-  Upload your own image in `.jpg`, `.jpeg`, or `.png` format
-  Custom-designed frontend using Streamlit + CSS
-  Clean modular code (`model.py`, `train.py`, `utils.py`, `app.py`)
-  Built with TensorFlow + Keras (saved `.h5` model)
-  Ready to deploy on Streamlit Cloud / Hugging Face Spaces

## 📁 Folder Structure

```
image-classification-cnn/
├── app.py               # Streamlit frontend
├── train.py             # Training script for the CNN
├── model.py             # CNN model architecture
├── utils.py             # Utility functions (plots, augmentations)
├── saved_model/
│   └── cifar10_model.h5 # Trained Keras model
├── README.md            # Project documentation
└── venv/                # (Optional) Python virtual environment
```
