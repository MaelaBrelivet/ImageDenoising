# Image Denoising with Convolutional Neural Networks (CNN)

## Overview
This project focuses on **image denoising using Convolutional Neural Networks (CNNs)** applied to real-world smartphone images.  
The goal is to remove noise while preserving important visual details such as textures, edges, and colors — a core challenge in **computational photography** and **computer vision**.

The project uses the **Smartphone Image Denoising Dataset (SIDD)** and implements a complete deep learning pipeline, from data preprocessing to qualitative and quantitative evaluation.

---

## Motivation
Smartphone images often suffer from significant noise, especially in **low-light conditions**.  
Traditional denoising algorithms tend to oversmooth images, while deep learning approaches can **learn noise distributions directly from data**, leading to better reconstruction quality.

---

## Key Features
- CNN-based denoising of RGB images
- Gaussian noise augmentation
- Custom training, validation, and test split
- Quantitative and qualitative evaluation
- Fully reproducible experiment in a single notebook

---

## Dataset
- **Name**: Smartphone Image Denoising Dataset (SIDD)
- **Source**: https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset
- Paired **noisy / ground-truth** images
- Images resized to **256×256**

---

## Model Architecture
- Sequential **Convolutional Neural Network**
- ReLU activations for hidden layers
- Sigmoid activation for output layer
- Optimizer: **Adam**
- Loss function: **Mean Squared Error (MSE)**

---

## Results

### Quantitative Results
The following metrics are computed on the test set:

- **Average PSNR**: **27.79 dB**
- **Average MSE**: **0.0020**

These results indicate a significant reduction of noise while maintaining image fidelity.

---

### Qualitative Results
Visual comparison between the **noisy input**, **CNN-denoised output**, and **ground-truth image**:

![Denoising Results](images/denoising_comparison.png)

The model effectively removes high-frequency noise while preserving the overall structure and colors of the images.  
Some smoothing of fine textures can still be observed, illustrating the trade-off between noise reduction and detail preservation.

---

## Technologies Used
- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **scikit-image**
- **tqdm**

---

## Installation & Usage

### 1. Install dependencies
```bash
pip install tensorflow opencv-python matplotlib numpy scikit-image tqdm
```

### 2. Run the notebook
The notebook covers:
- Dataset loading and preprocessing
- Noise augmentation
- CNN training
- Evaluation and visualization

## Skills Demonstrated
- Computer Vision
- Deep Learning with CNNs
- Image denoising
- Model evaluation (PSNR, MSE)
- Experimental analysis and visualization

## Author
Maëla Brelivet
Data Science Student @ EURECOM
