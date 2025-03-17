import torch
import torch.nn as nn

class Block_1(nn.Module):
    def __init__(self):
        super(Block_1, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))  # Reduced kernel size and stride

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        op1 = self.maxpool1(x)
        return op1


class Block_2_1(nn.Module):
    def __init__(self):
        super(Block_2_1, self).__init__()
        self.relu = nn.ReLU()
        self.dropout_percentage = 0.5

        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2_1_1 = nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout2_1 = nn.Dropout(p=self.dropout_percentage)

    def forward(self, op1):
        x = self.conv2_1_1(op1)
        x = self.batchnorm2_1_1(x)
        x = self.relu(x)
        x = self.conv2_1_2(x)
        x = self.batchnorm2_1_2(x)
        x = self.dropout2_1(x)
        op2_1 = self.relu(x + op1)
        return op2_1


class Block_2_2(nn.Module):
    def __init__(self):
        super(Block_2_2, self).__init__()
        self.relu = nn.ReLU()
        self.dropout_percentage = 0.5

        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2_2 = nn.Dropout(p=self.dropout_percentage)

    def forward(self, op2_1):
        x = self.conv2_2_1(op2_1)
        x = self.batchnorm2_2_1(x)
        x = self.relu(x)
        x = self.conv2_2_2(x)
        x = self.batchnorm2_2_2(x)
        x = self.dropout2_2(x)
        op2 = self.relu(x + op2_1)
        return op2


class Block_3(nn.Module):
    def __init__(self):
        super(Block_3, self).__init__()
        self.relu = nn.ReLU()
        self.dropout_percentage = 0.5

        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)

        # Skip connection
        self.concat_adjust_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0))
        self.dropout3_1 = nn.Dropout(p=self.dropout_percentage)

        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout3_2 = nn.Dropout(p=self.dropout_percentage)

    def forward(self, op2):
        x = self.conv3_1_1(op2)
        x = self.batchnorm3_1_1(x)
        x = self.relu(x)
        x = self.conv3_1_2(x)
        x = self.batchnorm3_1_2(x)
        x = self.dropout3_1(x)

        op2 = self.concat_adjust_3(op2)  # Skip connection
        op3_1 = self.relu(x + op2)

        x = self.conv3_2_1(op3_1)
        x = self.batchnorm3_2_1(x)
        x = self.relu(x)
        x = self.conv3_2_2(x)
        x = self.batchnorm3_2_2(x)
        x = self.dropout3_2(x)
        op3 = self.relu(x + op3_1)

        return op3


class Resnet18(nn.Module):
    def __init__(self, n_classes):
        """Model Architecture for Resnet18 model

        Args:
            n_classes (int): number of output classes
        """
        super(Resnet18, self).__init__()

        self.block1 = Block_1()
        self.block2_1 = Block_2_1()
        self.block2_2 = Block_2_2()
        self.block3 = Block_3()

        # Final block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Dynamically adjust pooling size
        self.fc = nn.Linear(in_features=128, out_features=1000)
        self.out = nn.Linear(in_features=1000, out_features=n_classes)

    def forward(self, x):
        op1 = self.block1(x)
        op2_1 = self.block2_1(op1)
        op2 = self.block2_2(op2_1)
        op3 = self.block3(op2)

        x = self.avgpool(op3)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.out(x)

        return x
    
    
    
    