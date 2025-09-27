# MNIST Digit Classification with CNNs

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify handwritten digits from the MNIST dataset. The codebase is modular, allowing for the sequential training and evaluation of three distinct CNN models with varying, lightweight architectures.

## 📜 Table of Contents

- [Project Overview](#-project-overview)
- [🚀 Technology Stack](#-technology-stack)
- [📁 File Structure](#-file-structure)
- [🧠 Understanding the CNN for MNIST](#-understanding-the-cnn-for-mnist)
- [🏗️ Model Architectures](#️-model-architectures)
- [⚙️ How to Run](#️-how-to-run)
- [📈 Results and Analysis](#-results-and-analysis)
- [🏆 Comparative Analysis](#-comparative-analysis)

## 📖 Project Overview

The primary goal of this project is to build, train, and evaluate lightweight CNN models for handwritten digit recognition. The models are intentionally designed to have a trainable parameter count between **3,500 and 7,500**, ensuring they are computationally efficient while maintaining high accuracy.

The project is structured into several modules, separating concerns like data loading, model definition, and the training process. This makes the code clean, readable, and easy to maintain.

## 🚀 Technology Stack

- **Python 3.x**
- **PyTorch**: For building and training the neural networks.
- **torchvision**: For accessing the MNIST dataset and image transformations.
- **matplotlib**: For plotting the training and test accuracy graphs.
- **uv**: As the package installer and virtual environment manager.

## 📁 File Structure

The project is organized into the following modules:


.
├── data_loader.py # Handles downloading, transforming, and loading the MNIST dataset.
├── model_v1.py # Defines the DS_CNN_V1 model class.
├── model_v2.py # Defines the DS_CNN_V2 model class.
├── model_v3.py # Defines the DS_CNN_V3 model class.
├── train.py # Contains the core training and testing logic.
└── main.py # Main script to orchestrate the training and evaluation of all models.




## 🧠 Understanding the CNN for MNIST

### What is a CNN?

A Convolutional Neural Network (CNN) is a class of neural networks particularly effective for analyzing visual imagery. Unlike standard networks, CNNs use a special operation called **convolution**, which allows them to automatically learn spatial hierarchies of features from images—from simple edges to complex shapes.

### The MNIST Dataset

The MNIST dataset is a classic collection of 70,000 grayscale images of handwritten digits (0-9). Each image is a small 28x28 pixel square. The dataset is split into 60,000 training images and 10,000 testing images.

### Key Neural Network Layers

Our models use a sequence of layers to process the images and learn to classify them.

> **`nn.Conv2d` (Convolutional Layer)**
> This is the core building block of a CNN. It works by sliding a small filter (kernel) over the input image, creating a "feature map." This process detects features like edges in early layers and more complex patterns in deeper layers.
>
> ```python
> # Example: conv1 = nn.Conv2d(1, 16, kernel_size=3)
> # in_channels=1: The input is a single-channel (grayscale) image.
> # out_channels=16: The layer produces 16 different feature maps.
> ```

> **`nn.MaxPool2d` (Max Pooling)**
> This layer reduces the spatial dimensions of the feature maps, making the feature detection more robust to the position of features in the image.

> **`nn.AdaptiveAvgPool2d` (Adaptive Average Pooling)**
> This layer reduces each feature map to a single number by taking the average. It guarantees a fixed-size output, which is a flexible way to prepare data for the final classification layer.

> **`nn.Linear` (Fully Connected Layer)**
> This is a standard neural network layer that takes the extracted features and performs the final classification. It outputs 10 values, representing the model's confidence for each digit (0-9).

## 🏗️ Model Architectures

### Model 1: `DS_CNN_V1`
- **Total Trainable Parameters: 5,226**
| Layer                  | Details                               |
| ---------------------- | ------------------------------------- |
| `Conv2d`               | 1 in channel, 16 out channels, 3x3 kernel |
| `MaxPool2d`            | 2x2 kernel                            |
| `Conv2d`               | 16 in channels, 32 out channels, 3x3 kernel |
| `MaxPool2d`            | 2x2 kernel                            |
| `AdaptiveAvgPool2d`    | Output size (1, 1)                    |
| `Linear`               | 32 in features, 10 out features       |

### Model 2: `DS_CNN_V2`
- **Total Trainable Parameters: 4,070**
| Layer                  | Details                               |
| ---------------------- | ------------------------------------- |
| `Conv2d`               | 1 in channel, 14 out channels, 3x3 kernel |
| `MaxPool2d`            | 2x2 kernel                            |
| `Conv2d`               | 14 in channels, 28 out channels, 3x3 kernel |
| `MaxPool2d`            | 2x2 kernel                            |
| `AdaptiveAvgPool2d`    | Output size (1, 1)                    |
| `Linear`               | 28 in features, 10 out features       |

### Model 3: `DS_CNN_V3`
- **Total Trainable Parameters: 6,526**
| Layer                  | Details                               |
| ---------------------- | ------------------------------------- |
| `Conv2d`               | 1 in channel, 18 out channels, 3x3 kernel |
| `MaxPool2d`            | 2x2 kernel                            |
| `Conv2d`               | 18 in channels, 36 out channels, 3x3 kernel |
| `MaxPool2d`            | 2x2 kernel                            |
| `AdaptiveAvgPool2d`    | Output size (1, 1)                    |
| `Linear`               | 36 in features, 10 out features       |

## ⚙️ How to Run

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Remo-Consultants/Session_6.git
    cd Session_6
    ```

2.  **Set up the environment:**
    Ensure you have `uv` installed. Create and sync the virtual environment with the required packages.
    ```sh
    # Create a virtual environment
    uv venv

    # Activate the environment
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate

    # Install dependencies
    uv pip install torch torchvision matplotlib
    ```

3.  **Run the main script:**
    Execute the `main.py` script to start the training and evaluation process for all three models.
    ```sh
    python main.py
    ```

## 📈 Results and Analysis

The following section details the performance of each model over 15 epochs of training.

### Model 1: `DS_CNN_V1` Results

![Model 1 Log](Images/Model_1_Logs.jpg)

**Performance Plot:**

![Model 1 Output](Images/Model_1_Output.png)

### Model 2: `DS_CNN_V2` Results

![Model 2 Log](Images/Model_2_Logs.jpg)

**Performance Plot:**
![Model 2 Output](Images/Model_2_Output.png)

Model 3: DS_CNN_V3 Results

![Model 3 Log](Images/Model_3_Logs.jpg)

**Performance Plot:**
![Model 3 Output](Images/Model_3_Output.png)

🏆 Comparative Analysis
This table summarizes the final performance of the three models, highlighting the trade-offs between parameter count and accuracy.
Model	Trainable Parameters	Final Test Accuracy (after 15 epochs)	Key Features
DS_CNN_V1	5,226	97.49%	Simplest architecture, baseline model
DS_CNN_V2	4,070	99.27%	High accuracy, efficient design
DS_CNN_V3	6,526	99.25%	Most parameters, excellent accuracy
Conclusion: All three lightweight models achieve impressive accuracy. Model 2 (DS_CNN_V2) provides a fantastic balance of high performance (99.27% accuracy) and a very low parameter count, making it arguably the most efficient of the three.
