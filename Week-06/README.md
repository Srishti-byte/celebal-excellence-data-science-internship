# Week 6 – Denoising Autoencoder using Convolutional Neural Networks

## Objective

The objective of this project is to implement a Convolutional Denoising Autoencoder capable of reconstructing clean handwritten digit images from noisy inputs. The model is trained on the MNIST dataset to learn meaningful feature representations for effective image denoising.

## Dataset

The project uses the **MNIST handwritten digit dataset**, which consists of grayscale images of handwritten digits (0–9).

Dataset Details:

- 60,000 Training Images
- 10,000 Test Images
- Image Size: 28 × 28 pixels
- Single-channel (Grayscale)

## Tasks Performed

- Data loading and preprocessing
- Image normalization and reshaping
- Artificial Gaussian noise generation
- Visualization of original and noisy images
- Design and implementation of a CNN-based Denoising Autoencoder
- Model compilation using Adam optimizer and Mean Squared Error (MSE) loss
- Model training using EarlyStopping callback
- Generation of denoised images
- Visualization of reconstructed images
- Training and validation loss analysis
- Experimental comparison using two Gaussian noise factors (0.3 and 0.2)
- Comparative analysis of reconstruction performance

## Results

- The denoising autoencoder successfully reconstructed clean handwritten digit images from noisy inputs.
- The reconstructed images effectively removed most of the Gaussian noise while preserving the overall structure of the handwritten digits.
- The training process converged successfully, achieving a final training loss of **0.0045** and validation loss of **0.0051** for the primary experiment.
- A comparative experiment demonstrated that reducing the Gaussian noise factor from **0.3** to **0.2** improved both reconstruction quality and reconstruction loss.
- The model demonstrated robust denoising performance across different Gaussian noise levels.

## Technologies Used

- Python
- NumPy
- Matplotlib
- TensorFlow
- Keras
- Google Colab

## Files

- **Week6_SrishtiGupta.ipynb** – Complete implementation of the Convolutional Denoising Autoencoder.
