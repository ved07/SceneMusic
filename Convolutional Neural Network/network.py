import torch
import torch.nn as nn
from torchsummary import summary
from dataHandling import dataloader


# Define Layers
def conv_layer(input_dim, output_dim,dropout=0.25):
    # Half the image size
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.ReLU(),
        nn.Dropout(dropout)
    )


def max_pool_layer():
    # add one to image size
    return nn.Sequential(
        nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1),padding=(1,1))
    )


def dense_layer(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Sigmoid()
    )


class Generator(nn.Module):
    def __init__(self,music_size,im_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            conv_layer(im_channels, 8),
            max_pool_layer(),
            conv_layer(8, 16),
            nn.Flatten(),
            dense_layer(16 * 38 * 38, 4096),
            dense_layer(4096, 256),
            dense_layer(256, 64),
            dense_layer(64, 8),

        )
        self.lstm = nn.LSTM(input_size=8, hidden_size=music_size, batch_first=True)


    def summarize(self):
        return summary(self.model,input_size=(3, 150, 150))


    def forward(self,x):
        x = self.model(x)
        x = x.reshape(x.shape[0],1,x.shape[1])
        x = self.lstm(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM =