# Brain Tumor Classification using Deep Learning

## Project Overview

This project is a deep learning-based medical image classification system for brain tumor diagnosis from MRI scans.

The model receives a brain MRI image and classifies it into one of three categories:

* Brain Glioma
* Brain Meningioma
* Brain Tumor

The system is implemented using PyTorch and Transfer Learning with the ResNet18 architecture.

---

## Dataset

Dataset Size:

* Total Images: 6056 MRI scans

Classes:

* brain_glioma (2004 images)
* brain_menin (2004 images)
* brain_tumor (2048 images)

Dataset Split:

* Training Set: 4237 images
* Validation Set: 907 images
* Test Set: 912 images

---

## Model Architecture

* ResNet18
* Transfer Learning
* Cross Entropy Loss
* Adam Optimizer
* Early Stopping

---

## Results

Validation Accuracy:

95.81%

Test Accuracy:

95.39%

Classification Performance:

| Class            | Precision | Recall | F1-Score |
| ---------------- | --------- | ------ | -------- |
| Brain Glioma     | 0.98      | 0.96   | 0.97     |
| Brain Meningioma | 0.92      | 0.97   | 0.95     |
| Brain Tumor      | 0.97      | 0.93   | 0.95     |

Overall Accuracy:

95.39%

---

## Project Structure

brain-tumor-ai/

├── data/

├── models/

├── results/

├── src/

│   ├── train.py

│   ├── evaluate.py

│   ├── predict.py

│   ├── confusion_matrix.py

│   └── split_dataset.py

├── README.md

└── requirements.txt

---

## Features

* Dataset Preparation
* Dataset Splitting
* Model Training
* Model Evaluation
* Brain Tumor Prediction
* Confusion Matrix Generation
* Model Saving and Loading

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Scikit-Learn

---

## Future Improvements

* Larger MRI datasets
* Data augmentation
* Mobile deployment
* Web application interface
* Explainable AI (Grad-CAM)

---

## Author

Sadaf Khalili
