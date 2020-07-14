import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch


class PuzzleDataset(Dataset):

    def __init__(self, path):
        self.directory = path
        self.image_label_list = self.read_file()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        img_path, label_path = self.image_label_list[index][0], self.image_label_list[index][1]
        img = Image.open(img_path)
        img = self.transforms(img)
        c = np.genfromtxt(label_path)
        return img, torch.from_numpy(c).float()

    def __len__(self):
        return len(self.image_label_list)

    def read_file(self):
        image_label_list = []
        imageDir = "{0}/images".format(self.directory)
        labelDir = "{0}/labels".format(self.directory)
        for file in os.listdir(imageDir):
            f = os.path.splitext(file)[0]
            image_label_list.append(
                (os.path.join(imageDir, "{0}.jpg".format(f)), os.path.join(labelDir, "{0}.txt".format(f))))
        return image_label_list
