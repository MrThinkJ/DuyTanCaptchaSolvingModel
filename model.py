import torch.nn as nn


class CaptchaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = self.make_block(1, 16, 3)
        self.block2 = self.make_block(16, 32, 3)
        self.max_pooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features=1152, out_features=512), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features=512, out_features=256), nn.ReLU())
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def make_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            # nn.BatchNorm2d(num_features=out_channels),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
