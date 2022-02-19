# Importing Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

CLASSES = {1:"Buildings", 2:"Forests", 3:"Mountains", 4:"Glacier", 5:"Street", 6:"Sea"}

FILENAME = 'newtrain.csv'
dataframe = pd.read_csv(FILENAME)


class Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 1])
        image = io.imread(img_name)
        label = self.dataframe.iloc[idx, 2]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample



class to_tensor(object):

    def __call__(self, sample):
        onehotvector = np.array([0 for x in range(6)])
        image,label = sample['image'],int(sample['label'])
        image = np.array(image)
        onehotvector[label] = 1
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(onehotvector)}


transformed_dataset = Dataset(csv_file='newtrain.csv',
                                           root_dir='train/',
                                           transform=transforms.Compose([
                                               to_tensor()
                                           ]))

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True)