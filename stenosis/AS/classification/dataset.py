import torch.utils.data as data
import os
import numpy as np
import torch
import pickle
import random
from .utils import *

class TwoViewDataset(data.Dataset):
    # 2views dataset: PLAX-2D & PSAX-A-2D for AS
    def __init__(self, path, transform, view_dropout=False, drop_rate=0.2, clip_size=64, random_clip=False):
        self.flist = pickle.load(open(path, 'rb'))  #list of (npyfolder, label)
        self.transform = transform
        self.view_dropout = view_dropout
        self.drop_rate = drop_rate
        self.random_clip = random_clip
        self.clip_size = clip_size

    def __getitem__(self, index):
        tmp = self.flist[index]
        folder = tmp[0]
        label = tmp[1]
        plax = self.get_view(folder, 'PLAX-2D')
        psax = self.get_view(folder, 'PSAX-A-2D')
        v1 = self.parse_video(plax)
        v2 = self.parse_video(psax)
        if self.view_dropout and len(plax)>0 and len(psax)>0:
            v1, v2 = self.drop_view_randomly(v1, v2)
        label = torch.from_numpy(np.array(label).astype('int64'))
        return v1, v2, label

    def drop_view_randomly(self, video1, video2):
        # randomly drop one view during training
        videos = [video1, video2]
        drop_index = random.choice([0, 1])
        if random.random() < self.drop_rate:
            videos[drop_index] *= 0
        return videos       

    def parse_video(self, npy_path):
        if len(npy_path) == 0:
            video = np.zeros((self.clip_size, 224, 224, 3)).astype('uint8')
        else:
            array = np.load(npy_path)
            array = remove_info(array)
            video = clip_video(array, size=self.clip_size, random_clip=self.random_clip)
            video = crop_resize_video(video, size=224)
        video = self.transform(video)
        return video

    def get_view(self, folder, view):
        files = os.listdir(folder)
        npy_file = ''
        for f in files:
            if (view+'.npy') in f:
                npy_file = os.path.join(folder, f)
                break
        return npy_file

    def __len__(self):
        return len(self.flist)
        