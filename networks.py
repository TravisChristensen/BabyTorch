import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 800)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(800, 800)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(800, 800)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(800, 27)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        x = self.drop3(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1_out = 32
        self.conv2_out = 64
        self.conv_final = self.conv2_out
        self.conv1 = nn.Conv2d(1, self.conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.conv1_out, self.conv2_out, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.conv_final * 7 * 7, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 27)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, self.conv_final * 7 * 7)  # 7 because image x / 2d pool makes 28/2/2 = 7
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x