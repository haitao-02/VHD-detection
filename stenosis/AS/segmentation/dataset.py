import os
import torch
import numpy as np
from PIL import Image
import pickle
from .utils import *


class CWDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, root, transform=None):
        self.nlist = pickle.load(open(file_list, 'rb')) # list of data index
        self.root = root
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.nlist[idx]) + '.png'
        mask_path = os.path.join(self.root, self.nlist[idx]) + '_mask.png'

        img = np.array(Image.open(img_path))  #[h,w,3], uint8
        mask = np.array(Image.open(mask_path)) #[h,w], 0 or 255
        img = crop_resize_image(img, size=512, convert_RGB=True)
        mask = crop_resize_image(mask, size=512, convert_RGB=False)
        mask = np.where(mask>127, 1, 0)
        mask = torch.tensor(mask).long()
        if self.transform:
            img = self.transform(img)
        return img, mask
    
    def __len__(self):
        return len(self.nlist)


