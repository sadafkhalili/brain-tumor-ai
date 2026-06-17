# Brain Tumor Classification using Deep Learning

## Project Overview

This project uses Deep Learning and Convolutional Neural Networks (CNNs) to classify brain MRI images into different tumor categories.

The model is based on ResNet18 and is implemented using PyTorch.

## Dataset

The project is designed to work with brain MRI images organized into three classes:

* Glioma
* Meningioma
* Tumor

Dataset structure:

data/raw/

├── Glioma/

├── Meningioma/

└── Tumor/

## Project Structure

brain-tumor-ai/

├── data/

├── models/

├── notebooks/

├── src/

│   ├── config.py

│   ├── dataset.py

│   ├── split_dataset.py

│   ├── train.py

│   ├── evaluate.py

│   └── predict.py

└── README.md

## Features

* Automatic dataset splitting
* Data loading using PyTorch DataLoader
* Transfer Learning with ResNet18
* Validation accuracy monitoring
* Best model saving
* Model evaluation
* Prediction on new MRI images

## Training Pipeline

1. Place dataset in data/raw
2. Run split_dataset.py
3. Run train.py
4. Run evaluate.py
5. Run predict.py

## Model

* Architecture: ResNet18
* Framework: PyTorch
* Input Size: 224×224
* Output Classes: 3

## Future Improvements

* More MRI datasets
* Data augmentation
* Early stopping
* Confusion matrix
* Grad-CAM visualization
* Deployment using Flask or Streamlit

## Author

Sadaf Khalili
