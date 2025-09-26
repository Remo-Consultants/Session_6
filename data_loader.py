# data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_mnist_data_loaders(batch_size, train_subset_size=50000, test_subset_size=10000):
    """
    Loads and preprocesses the MNIST dataset, returning training and testing data loaders.

    Args:
        batch_size (int): The batch size for the data loaders.
        train_subset_size (int): The number of samples to use for the training set.
        test_subset_size (int): The number of samples to use for the testing set.

    Returns:
        tuple: A tuple containing (train_loader, test_loader).
    """
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])

    full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

    train_dataset = Subset(full_train_dataset, range(train_subset_size))
    test_dataset = Subset(full_test_dataset, range(test_subset_size))

    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Testing Dataset: {len(test_dataset)}")
    print(f"Batch Size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # Example usage:
    batch_size = 32
    train_loader, test_loader = get_mnist_data_loaders(batch_size)
    
    # Verify a batch
    for data, target in train_loader:
        print(f"Batch data shape: {data.shape}")
        print(f"Batch target shape: {target.shape}")
        break