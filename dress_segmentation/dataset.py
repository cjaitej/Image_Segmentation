import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Resize

class DressDataset(Dataset):
    def __init__(self, image_directory, mask_directory):
        self.size = (256, 128)
        self.image_dir = image_directory
        self.mask_dir = mask_directory
        self.data = os.listdir(self.mask_dir)
        x = int(len(self.data)*0.7)
        self.train_data = self.data[:x]
        self.test_data = self.data[x:]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index, train=True):
        if train == False:
            self.train_data = self.test_data
        img_path = os.path.join(self.image_dir, self.train_data[index])
        mask_path = os.path.join(self.mask_dir, self.train_data[index])
        image = np.array(Image.open(img_path).convert("RGB").resize(self.size))/255.0
        mask = np.array(Image.open(mask_path).resize((self.size))).astype('int8')/255.0

        image = image.reshape((3, self.size[1], self.size[0]))
        mask = mask.reshape((1, self.size[1], self.size[0]))
        return image, mask



