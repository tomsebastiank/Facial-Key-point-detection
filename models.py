import torch
import torch.nn as neural_network
import torch.nn.functional as function
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(neural_network.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Convolution Layers, Total 4
        # For the symmetry of kernels ,all kernals size as taken as odd numbers 3
        self.conv1 = neural_network.Conv2d(1, 32, 3)
        self.conv2 = neural_network.Conv2d(32, 64, 3)
        self.conv3 = neural_network.Conv2d(64, 128, 3)
        self.conv4 = neural_network.Conv2d(128, 128, 3)

        # Maxpool layer, it is repeatedly used, kernal size 2 and stride is 2
        self.pool = neural_network.MaxPool2d(2, 2)

        # Fully Connected Layers
        # Total image size in the output is 12*12 pixel, so after flattening total number of features
        # will be 12*12*128 = 18432  (Total 128 channels )
        self.fc_1 = neural_network.Linear(18432, 1000)
        # As a rule of thumb total number of neurons should be between 136 and 1000, so 700 is chosen
        self.fc_2 = neural_network.Linear(1000, 1000)
        # Total 68 keypoints , so total number of output is 68*2 (x & y)
        self.fc_3 = neural_network.Linear(1000, 136)

        # Dropouts
        self.drop1 = neural_network.Dropout(p = 0.1)
        self.drop2 = neural_network.Dropout(p = 0.2)
        self.drop3 = neural_network.Dropout(p = 0.3)
        self.drop4 = neural_network.Dropout(p = 0.4)
        # Same layer is used multiple times
        self.drop5 = neural_network.Dropout(p = 0.5)
        self.drop6 = neural_network.Dropout(p = 0.7)

    def forward(self, x):

        # Cascaded operations convolution ---> relu activation ---> max pool --> drop out
        x_1 = self.drop1(function.relu(self.pool (self.conv1(x))))
        x_2 = self.drop2(function.relu(self.pool(self.conv2(x_1))))
        x_3 = self.drop3(function.relu(self.pool(self.conv3(x_2))))
        x_4 = self.drop4(function.relu(self.pool(self.conv4(x_3))))

        # Reshaping of the 3d array
        x_5 = x_4.view(x_4.size(0), -1)

        x_6 = self.drop5(function.relu(self.fc_1(x_5)))

        x_7 = self.drop6(function.relu(self.fc_2(x_6)))

        x_8 = self.fc_3(x_7)

        return x_8