import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import get_mnist_data_loaders
from model_v1 import DS_CNN_V1
from model_v2 import DS_CNN_V2
from model_v3 import DS_CNN_V3
from train import train, test, plot_accuracies

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    epochs = 15
    learning_rate = 0.03

    # Data loaders
    train_loader, test_loader = get_mnist_data_loaders(batch_size)

    # List of models to run
    models_to_run = {
        "DS_CNN_V1": DS_CNN_V1,
        "DS_CNN_V2": DS_CNN_V2,
        "DS_CNN_V3": DS_CNN_V3,
    }

    for model_name, ModelClass in models_to_run.items():
        print(f"\n--- Running Model: {model_name} ---")
        
        model = ModelClass().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=0, factor=0.5)

        # Verify parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters for {model_name}: {total_params}")
        
        if total_params < 3500 or total_params > 7500:
            print(f"Warning: Total trainable parameters for {model_name} ({total_params}) are not within the 3500-7500 range. Please adjust model architecture.")
            # For now, we'll continue to demonstrate sequential execution
            # You might want to 'continue' here to skip training problematic models
            
        # Training loop
        train_accuracies = []
        test_accuracies = []
        train_count = len(train_loader.dataset)
        test_count = len(test_loader.dataset)

        for epoch in range(epochs):
            train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
            test_loss, test_acc = test(model, device, test_loader, criterion)
            scheduler.step(test_acc)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print(f"Epoch {epoch+1}: "
                  f"Train Acc: {train_acc:.2f}% (Train set: {train_count}), "
                  f"Test Acc: {test_acc:.2f}% (Test set: {test_count})")

        # Plotting results
        plot_accuracies(train_accuracies, test_accuracies, epochs)
        plt.close() # Close the plot after each model to avoid overlapping figures

if __name__ == '__main__':
    main()