# importing libraries
from torchsummary import summary
import torch.nn as nn

""" Converting the images to a class so that we can concatenate this onto the inputs, """



def conv_layer(input_dim, output_dim):
    # Half the image size
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.ReLU()
    )


def max_pool_layer():
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1),padding=(1,1))
    )


def dense_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Sigmoid()
    )


class Network(nn.Module):




    def __init__(self, im_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            conv_layer(im_channels, 16),
            max_pool_layer(),
            conv_layer(16, 32),
            conv_layer(32,64),
            max_pool_layer(),
            nn.Flatten(),
            dense_layer(20*20*64,256),
            dense_layer(256,64),
            dense_layer(64,6),
            nn.Softmax()
        )


    def Summary(self):
        return summary(self.model, input_size=(3, 150, 150))


    def forward(self, x):
        return self.model(x)

net = Network()
print(net.Simummary())