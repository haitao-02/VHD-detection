import torch.utils.data as data
import numpy as np
import torch
import pickle


class CWPoint_dataset(data.Dataset):
    """
    deal with processed imgs and heatmaps
    img shape: [256, 256, 3]
    heatmap: [2, 64, 64]
    one img contains one wave
    """
    def __init__(self, path, transform=None):
        self.data_list = pickle.load(open(path, 'rb'))   #list of (img, heatmap)
        self.transform = transform

    def __getitem__(self, idx):
        tmp = self.data_list[idx]
        img = np.load(tmp[0])
        hm = np.load(tmp[1])
        img = img.astype('uint8')
        if self.transform is not None:
            img = self.transform(img)
        hm = torch.from_numpy(hm)
        return img, hm

    def __len__(self):
        return len(self.data_list)
