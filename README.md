# Image Denoising with CNN

## Description
This project implements an image denoising pipeline using **Convolutional Neural Networks (CNNs)**. The goal is to remove noise from smartphone images while preserving image details. The dataset used is the **Smartphone Image Denoising Dataset (SIDD)**.

The project is implemented in **Python** using **TensorFlow/Keras**, **OpenCV**, and **NumPy**.

## Features
- Denoising of noisy RGB images using a CNN.
- Gaussian noise augmentation.
- Training, validation, and testing split.
- Evaluation with **PSNR** and **MSE** metrics.
- Visualization of noisy, denoised, and original images.

## Dataset
- Source: [Smartphone Image Denoising Dataset (SIDD)](https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset)
- Dataset contains pairs of noisy and ground-truth images.
- Images are resized to **256x256** for training.

## Model
- Sequential CNN with multiple convolutional layers.
- ReLU activations for hidden layers, Sigmoid for output.
- Optimizer: Adam.
- Loss: Mean Squared Error (MSE).

## Usage

1. **Install dependencies**:

```bash
pip install tensorflow opencv-python matplotlib numpy scikit-image tqdm
```

2. **Run the notebook**:
- Load and preprocess the dataset.
- Train the CNN on noisy vs. original images.
- Evaluate on test images.
- Visualize results using matplotlib.

## Results
- Average PSNR and MSE are calculated on the test set.
- Sample results show noisy, denoised, and original images side by side for comparison.

## Author
Maëla Brelivet – Data Science Student, EURECOM
