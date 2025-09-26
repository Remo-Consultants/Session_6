import torch
import torch.nn as nn

class DS_CNN_V2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(1, 4, kernel_size=3, padding=0)  # Output 26x26x4
        self.bn1_1 = nn.BatchNorm2d(4)

        self.conv1_2 = nn.Conv2d(4, 4, kernel_size=3, padding=0)  # Output 24x24x4
        self.bn1_2 = nn.BatchNorm2d(4)

        self.conv1_3 = nn.Conv2d(4, 8, kernel_size=3, padding=0)  # Output 22x22x8
        self.bn1_3 = nn.BatchNorm2d(8)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 11x11x8
        self.dropout1 = nn.Dropout(0.1)

        # Block 2
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=0)  # Output 9x9x16
        self.bn2_1 = nn.BatchNorm2d(16)

        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=0)  # Output 7x7x16
        self.bn2_2 = nn.BatchNorm2d(16)

        self.conv2_3 = nn.Conv2d(16, 16, kernel_size=3, padding=0)  # Output 5x5x16
        self.bn2_3 = nn.BatchNorm2d(16)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 2x2x16
        self.dropout2 = nn.Dropout(0.2)

        # Classification Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output 16 features
        self.fc = nn.Linear(in_features=16, out_features=10)  # Output 10 classes

    def forward(self, x):
        x = nn.ReLU()(self.bn1_1(self.conv1_1(x)))
        x = nn.ReLU()(self.bn1_2(self.conv1_2(x)))
        x = nn.ReLU()(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = nn.ReLU()(self.bn2_1(self.conv2_1(x)))
        x = nn.ReLU()(self.bn2_2(self.conv2_2(x)))
        x = nn.ReLU()(self.bn2_3(self.conv2_3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v2 = DS_CNN_V2().to(device)
    total_params_v2 = sum(p.numel() for p in model_v2.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters for DS_CNN_V2: {total_params_v2}")
