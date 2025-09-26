MNIST Digit Classification with Convolutional Neural Networks
This project demonstrates the use of Convolutional Neural Networks (CNNs) to classify handwritten digits from the MNIST dataset. The codebase is modular, allowing for the sequential training and evaluation of three distinct CNN models with varying architectures.
Table of Contents
Project Overview
Technolog Stack
File Structure
Understanding the CNN for MNIST
What is a CNN?
The MNIST Dataset
Key Neural Network Layers
Model Architectures
Model 1: `DS_CNN_V1`
Model 2: `DS_CNN_V2`
Model 3: `DS_CNN_V3`
How to Run
Results and Analysis
Project Overview
The primary goal of this project is to build, train, and evaluate lightweight CNN models for the task of handwritten digit recognition. The models are designed to have a trainable parameter count between 3,500 and 7,500, ensuring they are computationally efficient while maintaining high accuracy.
The project is structured into several modules, separating concerns like data loading, model definition, and the training process, making the code clean, readable, and easy to maintain.
Technology Stack
Python 3.x
PyTorch: For building and training the neural networks.
torchvision: For accessing the MNIST dataset and image transformations.
matplotlib: For plotting the training and test accuracy graphs.
uv: As the package installer and virtual environment manager.
File Structure
The project is organized into the following modules:
data_loader.py: A module responsible for downloading, transforming, and loading the MNIST dataset into batches for training and testing.
model_v1.py, model_v2.py, model_v3.py: Each file contains a separate CNN model class (DS_CNN_V1, DS_CNN_V2, DS_CNN_V3) with a unique architecture.
train.py: Contains the core logic for the training and testing loops (train, test) and a function for plotting results (plot_accuracies).
main.py: The main entry point of the application. It orchestrates the entire process by importing the necessary modules, instantiating the models, and running the training and evaluation sequence for each one.
Understanding the CNN for MNIST
What is a CNN?
A Convolutional Neural Network (CNN or ConvNet) is a class of neural networks that is particularly effective for analyzing visual imagery. Unlike standard neural networks, CNNs use a special operation called convolution, which allows them to automatically and adaptively learn spatial hierarchies of features from input images.
The MNIST Dataset
The MNIST dataset is a classic collection of 70,000 grayscale images of handwritten digits (0 through 9). Each image is a small square of 28x28 pixels. The dataset is split into 60,000 training images and 10,000 testing images.
Key Neural Network Layers
Our models use a sequence of layers to process the images and learn to classify them.
nn.Conv2d (Convolutional Layer): This is the core building block of a CNN. It works by sliding a small filter, called a kernel, over the input image. The kernel performs a dot product with the part of the image it's currently on, creating a "feature map." This process helps detect simple features like edges and corners in the early layers, and more complex features like shapes and patterns in deeper layers.
In conv1 = nn.Conv2d(1, 16, ...):
in_channels=1: The input is a single-channel (grayscale) image.
out_channels=16: The layer will produce 16 different feature maps, each corresponding to a unique learned kernel.
nn.BatchNorm2d (Batch Normalization): This layer is used to stabilize and accelerate the training process by normalizing the activations of the previous layer.
nn.MaxPool2d (Max Pooling): This layer reduces the spatial dimensions (height and width) of the feature maps. It works by taking the maximum value over a small window, which helps to make the feature detection more robust to changes in the position of features in the image.
nn.Dropout: A regularization technique used to prevent overfitting. During training, it randomly sets a fraction of input units to 0 at each update, which helps to prevent co-adaptation of neurons.
nn.AdaptiveAvgPool2d (Adaptive Average Pooling): This layer reduces each feature map to a single number by taking the average. Its "adaptive" nature means it will always produce a fixed-size output (in our case, 1x1) regardless of the input size, which is a flexible way to prepare the data for the final classification layer.
nn.Linear (Fully Connected Layer): This is a standard neural network layer where each neuron is connected to every neuron in the previous layer. It takes the features extracted by the convolutional layers and uses them to perform the final classification. In our case, it outputs 10 values, representing the model's confidence for each digit (0-9).
Model Architectures
Model 1: DS_CNN_V1
Total Trainable Parameters: 5,226
Architecture:
Conv2d (1 input channel, 16 output channels, 3x3 kernel)
MaxPool2d
Conv2d (16 input channels, 32 output channels, 3x3 kernel)
MaxPool2d
AdaptiveAvgPool2d
Linear (32 input features, 10 output features)
Model 2: DS_CNN_V2
Total Trainable Parameters: 4,070
Architecture:
Conv2d (1 input channel, 14 output channels, 3x3 kernel)
MaxPool2d
Conv2d (14 input channels, 28 output channels, 3x3 kernel)
MaxPool2d
AdaptiveAvgPool2d
Linear (28 input features, 10 output features)
Model 3: DS_CNN_V3
Total Trainable Parameters: 6,526
Architecture:
Conv2d (1 input channel, 18 output channels, 3x3 kernel)
MaxPool2d
Conv2d (18 input channels, 36 output channels, 3x3 kernel)
MaxPool2d
AdaptiveAvgPool2d
Linear (36 input features, 10 output features)
How to Run
Setup Environment: Ensure you have uv installed. Create and sync the virtual environment with the required packages.
Run the Main Script: Execute the main.py script to start the training and evaluation process for all three models.
Results and Analysis
The following section details the performance of each model over 19 epochs of training.
Model 1: DS_CNN_V1 Results
Training Log:
Using device: cpu
Training Dataset: 50000
Testing Dataset: 10000
Batch Size: 32

--- Running Model: DS_CNN_V1 ---
Total Trainable Parameters for DS_CNN_V1: 5226
Epoch 1: Train Acc: 73.79% (Train set: 50000), Test Acc: 93.78% (Test set: 10000)
Epoch 2: Train Acc: 88.67% (Train set: 50000), Test Acc: 94.60% (Test set: 10000)
Epoch 3: Train Acc: 90.59% (Train set: 50000), Test Acc: 96.34% (Test set: 10000)
Epoch 4: Train Acc: 91.70% (Train set: 50000), Test Acc: 96.09% (Test set: 10000)
Epoch 5: Train Acc: 93.18% (Train set: 50000), Test Acc: 96.85% (Test set: 10000)
Epoch 6: Train Acc: 93.19% (Train set: 50000), Test Acc: 97.49% (Test set: 10000)
Epoch 7: Train Acc: 93.65% (Train set: 50000), Test Acc: 96.41% (Test set: 10000)
Epoch 8: Train Acc: 94.21% (Train set: 50000), Test Acc: 97.29% (Test set: 10000)
Epoch 9: Train Acc: 94.30% (Train set: 50000), Test Acc: 97.52% (Test set: 10000)
Epoch 10: Train Acc: 94.52% (Train set: 50000), Test Acc: 97.31% (Test set: 10000)
Epoch 11: Train Acc: 94.76% (Train set: 50000), Test Acc: 97.48% (Test set: 10000)
Epoch 12: Train Acc: 94.89% (Train set: 50000), Test Acc: 97.37% (Test set: 10000)
Epoch 13: Train Acc: 94.79% (Train set: 50000), Test Acc: 97.57% (Test set: 10000)
Epoch 14: Train Acc: 94.86% (Train set: 50000), Test Acc: 97.48% (Test set: 10000)
Epoch 15: Train Acc: 94.92% (Train set: 50000), Test Acc: 97.49% (Test set: 10000)
Performance Plot:
!Model 1 Output
Model 2: DS_CNN_V2 Results
Training Log:
--- Running Model: DS_CNN_V2 ---
Total Trainable Parameters for DS_CNN_V2: 6590
Epoch 1: Train Acc: 87.89% (Train set: 50000), Test Acc: 96.79% (Test set: 10000)
Epoch 2: Train Acc: 93.49% (Train set: 50000), Test Acc: 97.96% (Test set: 10000)
Epoch 3: Train Acc: 94.47% (Train set: 50000), Test Acc: 97.99% (Test set: 10000)
Epoch 4: Train Acc: 94.95% (Train set: 50000), Test Acc: 98.58% (Test set: 10000)
Epoch 5: Train Acc: 95.16% (Train set: 50000), Test Acc: 98.73% (Test set: 10000)
Epoch 6: Train Acc: 95.40% (Train set: 50000), Test Acc: 98.42% (Test set: 10000)
Epoch 7: Train Acc: 96.44% (Train set: 50000), Test Acc: 98.69% (Test set: 10000)
Epoch 8: Train Acc: 97.16% (Train set: 50000), Test Acc: 99.16% (Test set: 10000)
Epoch 9: Train Acc: 97.03% (Train set: 50000), Test Acc: 99.19% (Test set: 10000)
Epoch 10: Train Acc: 97.10% (Train set: 50000), Test Acc: 99.17% (Test set: 10000)
Epoch 11: Train Acc: 97.26% (Train set: 50000), Test Acc: 99.19% (Test set: 10000)
Epoch 12: Train Acc: 97.43% (Train set: 50000), Test Acc: 99.23% (Test set: 10000)
Epoch 13: Train Acc: 97.52% (Train set: 50000), Test Acc: 99.29% (Test set: 10000)
Epoch 14: Train Acc: 97.56% (Train set: 50000), Test Acc: 99.24% (Test set: 10000)
Epoch 15: Train Acc: 97.57% (Train set: 50000), Test Acc: 99.27% (Test set: 10000)
Performance Plot:
!Model 2 Output
Model 3: DS_CNN_V3 Results
Training Log:
--- Running Model: DS_CNN_V3 ---
Total Trainable Parameters for DS_CNN_V3: 4538
Epoch 1: Train Acc: 88.02% (Train set: 50000), Test Acc: 97.51% (Test set: 10000)
Epoch 2: Train Acc: 93.69% (Train set: 50000), Test Acc: 97.81% (Test set: 10000)
Epoch 3: Train Acc: 94.28% (Train set: 50000), Test Acc: 98.37% (Test set: 10000)
Epoch 4: Train Acc: 94.84% (Train set: 50000), Test Acc: 98.67% (Test set: 10000)
Epoch 5: Train Acc: 95.23% (Train set: 50000), Test Acc: 98.67% (Test set: 10000)
Epoch 6: Train Acc: 96.20% (Train set: 50000), Test Acc: 99.09% (Test set: 10000)
Epoch 7: Train Acc: 96.20% (Train set: 50000), Test Acc: 98.89% (Test set: 10000)
Epoch 8: Train Acc: 96.68% (Train set: 50000), Test Acc: 99.12% (Test set: 10000)
Epoch 9: Train Acc: 96.68% (Train set: 50000), Test Acc: 99.02% (Test set: 10000)
Epoch 10: Train Acc: 96.93% (Train set: 50000), Test Acc: 99.14% (Test set: 10000)
Epoch 11: Train Acc: 96.92% (Train set: 50000), Test Acc: 99.20% (Test set: 10000)
Epoch 12: Train Acc: 97.06% (Train set: 50000), Test Acc: 99.10% (Test set: 10000)
Epoch 13: Train Acc: 96.97% (Train set: 50000), Test Acc: 99.25% (Test set: 10000)
Epoch 14: Train Acc: 97.10% (Train set: 50000), Test Acc: 99.24% (Test set: 10000)
Epoch 15: Train Acc: 97.20% (Train set: 50000), Test Acc: 99.25% (Test set: 10000)
Performance Plot:
!Model 3 Output