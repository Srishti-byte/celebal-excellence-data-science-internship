# Week 4 – Image Classification using Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN)

## Objective

The objective of this project is to build and compare Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for image classification using the CIFAR-10 dataset. The project also explores data augmentation and EarlyStopping techniques to improve model generalization and reduce overfitting.

## Dataset

The project uses the **CIFAR-10** image classification dataset, containing:

- 50,000 training images
- 10,000 testing images
- 10 object categories
- RGB images of size 32 × 32 pixels

## Tasks Performed

- Loaded and preprocessed the CIFAR-10 dataset
- Normalized image pixel values
- Built a baseline Artificial Neural Network (ANN)
- Built a Convolutional Neural Network (CNN)
- Compared ANN and CNN performance
- Plotted validation accuracy comparison
- Increased ANN layer configuration
- Updated CNN filters from **32 → 64 → 128**
- Increased training epochs from **10 to 20**
- Implemented Data Augmentation using RandomFlip, RandomRotation, and RandomZoom
- Applied EarlyStopping to prevent overfitting
- Built and evaluated an Enhanced CNN model
- Compared the performance of ANN, Baseline CNN, and Enhanced CNN

## Results

- CNN significantly outperformed the ANN on the CIFAR-10 image classification task.
- The Baseline CNN achieved **71.20%** test accuracy.
- The Enhanced CNN achieved **68.99%** test accuracy with the lowest test loss (**0.9044**).
- Data Augmentation and EarlyStopping improved the model's generalization while reducing the risk of overfitting.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Google Colab

## Files

- **Week4_SrishtiGupta.ipynb** – Complete implementation of ANN, CNN, Enhanced CNN, model evaluation, and performance comparison.
