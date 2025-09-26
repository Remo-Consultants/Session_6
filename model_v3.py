# model_v3.py
import torch
import torch.nn as nn

class DS_CNN_V3(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: Feature Extraction
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=0) # Output 26x26x4
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0) # Output 24x24x8
        self.bn2 = nn.BatchNorm2d(8)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output 12x12x8
        self.dropout1 = nn.Dropout(0.1)

        # Block 2: Further Feature Extraction
        self.conv3 = nn.Conv2d(8, 12, kernel_size=3, padding=0) # Output 10x10x12
        self.bn3 = nn.BatchNorm2d(12)

        self.conv4 = nn.Conv2d(12, 12, kernel_size=3, padding=0) # Output 8x8x12
        self.bn4 = nn.BatchNorm2d(12)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output 4x4x12
        self.dropout2 = nn.Dropout(0.2)

        # Block 3: Classification Head
        self.conv5 = nn.Conv2d(12, 16, kernel_size=3, padding=0) # Output 2x2x16
        self.bn5 = nn.BatchNorm2d(16)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # Output 16 features

        self.fc = nn.Linear(in_features=16, out_features=10) # Output 10 classes

    def forward(self, x):
        # Block 1
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten for the fully connected layer
        x = self.fc(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v3 = DS_CNN_V3().to(device)
    total_params_v3 = sum(p.numel() for p in model_v3.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters for DS_CNN_V3: {total_params_v3}")