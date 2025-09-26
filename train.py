# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train(model, device, train_loader, optimizer, criterion):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to train on (cpu or cuda).
        train_loader (DataLoader): The data loader for the training set.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: A tuple containing (average_loss, accuracy).
    """
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test(model, device, test_loader, criterion):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): The neural network model.
        device (torch.device): The device to evaluate on (cpu or cuda).
        test_loader (DataLoader): The data loader for the testing set.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: A tuple containing (test_loss, accuracy).
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def plot_accuracies(train_accuracies, test_accuracies, epochs):
    """
    Plots the training and test accuracies over epochs.

    Args:
        train_accuracies (list): List of training accuracies per epoch.
        test_accuracies (list): List of test accuracies per epoch.
        epochs (int): Total number of epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy over Epochs')
    plt.xticks(range(1, epochs + 1))
    plt.yticks([i for i in range(80, 101, 2)])
    plt.legend()
    plt.grid(True)
    plt.show()
