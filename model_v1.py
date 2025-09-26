# model_v1.py
import torch
import torch.nn as nn

class DS_CNN_V1(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # Output 28x28x16
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output 14x14x16
        self.dropout1 = nn.Dropout(0.1)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Output 14x14x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output 7x7x32
        self.dropout2 = nn.Dropout(0.2)

        # Classification Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Output 32 features
        self.fc = nn.Linear(in_features=32, out_features=10) # Output 10 classes

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v1 = DS_CNN_V1().to(device)
    total_params_v1 = sum(p.numel() for p in model_v1.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters for DS_CNN_V1: {total_params_v1}")