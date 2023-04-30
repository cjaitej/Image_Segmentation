import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torch import zeros, as_tensor, int8
from torchvision.transforms import Resize

class CarvanaDataset(Dataset):
    def __init__(self, image_directory, mask_directory, transform=None):
        self.size = (256, 512)#, 256)
        self.image_dir = image_directory
        self.mask_dir = mask_directory
        self.transform = transform
        self.masks = os.listdir(self.mask_dir)
        x = int(len(self.masks)*0.7)
        self.train_masks = self.masks[:x]
        self.test_masks = self.masks[x:]
        self.resize = Resize((512, 256))
        #Maybe this is unnecessary if the names of the mask can be derived from the image names list.

    def __len__(self):
        return len(self.train_masks)

    def __getitem__(self, index, train=True):
        if train == False:
            self.train_masks = self.test_masks
        img_path = os.path.join(self.image_dir, self.train_masks[index].replace("_segm.png", ".jpg"))
        mask_path = os.path.join(self.mask_dir, self.train_masks[index])
        image = np.array(Image.open(img_path).convert("RGB").resize(self.size))/255.0
        temp = np.array(Image.open(mask_path).resize((self.size)))
        mask = zeros((24, self.size[1], self.size[0]))

        for i in range(mask.shape[0]):
            mask[i] = as_tensor((temp == i), dtype= int8)
        # image = image.reshape()

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = image.reshape((3, self.size[1], self.size[0]))
        return image, mask



