# Importing Libraries
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import io

CLASSES = {1:"Buildings", 2:"Forests", 3:"Mountains", 4:"Glacier", 5:"Street", 6:"Sea"}

FILENAME = 'train.csv'
dataframe = pd.read_csv(FILENAME)


def load_data(data, classes):
    r = len(data)
    # r = 4
    to_drop = []
    for n in range(r-1,-1,-1):

        image_name = data.iloc[n, 0]
        integer_label = int(data.iloc[n, 1]) + 1
        # print(integer_label)
        image_class = classes[integer_label]
        # print("File: {}; Class {}".format(image_name, image_class))
        image = io.imread(os.path.join('train/',image_name))
        # Show images on matplotlib
        # plt.imshow(image)
        # plt.show()
        image = torch.from_numpy(np.array(image))

        if image.size()[0] != 150 or image.size()[1] !=150:
            to_drop.append(image_name)
            print(" {} has size is {}".format(image_name,image.size()))
    for item in to_drop:
        index = list(data["image_name"]).index(item)
        data = data.drop(index)
    return data

data = load_data(dataframe,CLASSES)
print(data)
data.to_csv('newtrain.csv')