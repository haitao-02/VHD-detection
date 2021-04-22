import os
import random
import torch
import numpy as np
import glob
import pickle
import cv2
cv2.setNumThreads(0)
from torch.utils.data import Dataset, DataLoader
from transforms import GroupRandomRescale, GroupResize, GroupRotate, GroupCentralCrop
from torchvision import transforms

def cv2_loader(path_list):
    video = np.asarray([cv2.imread(p) for p in path_list]).astype('float32')
    video[..., [0,2]] = video[..., [2,0]]
    return video

class RgDataset(Dataset):
    def __init__(self, data_list, win_size=32, transform=None, mode='train'):
        self.num_classes = 3
        self.data_list = data_list
        self.loader = cv2_loader
        self.win_size = win_size
        self.transform = transform
        self.mode = mode
    
    def __len__(self):
        return(len(self.data_list))
            
    def __getitem__(self, idx):
        sample = {}
        win_size = self.win_size
        data = self.data_list[idx]
        folder_path = data['folder_path']
        start = data['start']
        end = data['end']
        seg_idx = data['seg_idx']
            
        if start+win_size == end:
            img_list = np.array([os.path.join(folder_path, 'img_'+str(i).zfill(3)+'.png') for i in range(start, end)], dtype=np.unicode_)
            video = self.loader(img_list)
            kf= np.zeros(win_size, dtype='bool')
            kf[seg_idx-start] = 1
        else:
            img_list = np.array([os.path.join(folder_path, 'img_'+str(i).zfill(3)+'.png') for i in range(start, end)], dtype=np.unicode_)
            video = self.loader(img_list)
            # random zero padding
            l = random.randint(0, win_size-len(video)) if self.mode == 'train' else 0
            video = np.pad(video, ((l, win_size-l-(end-start)),(0,0),(0,0),(0,0)), mode='constant')
            kf = np.zeros(win_size, dtype='bool')
            kf[seg_idx-start+l] = 1
            
        mask = np.zeros((win_size, video.shape[1], video.shape[2], self.num_classes), dtype='float32')
        anno_list = np.array([os.path.join(folder_path, 'mask_'+str(i).zfill(3)+'.png') for i in seg_idx], dtype=np.unicode_)
        anno = self.loader(anno_list)
        anno = np.array(anno, dtype='float32') / 255
        mask[kf] = anno

        sample['video'] = video
        sample['mask'] = mask
        sample['kf'] = kf

        if self.transform:
            sample = self.transform(sample)
        return(sample)

class Normalize:
    def __call__(self, sample):
        sample['video'] = sample['video'] / 255
        return(sample)

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample['video'] = torch.from_numpy(sample['video']).permute(0,3,1,2).float()
        sample['mask'] = torch.from_numpy(sample['mask']).permute(0,3,1,2).float()
        sample['label'] = torch.from_numpy(sample['label']).long()
        sample['kf'] = torch.from_numpy(sample['kf']).byte()
        return (sample)

class RgLoader(DataLoader):
    def __init__(self, dataset_path, batch_size, win_size, num_workers, **kwargs):
        with open(dataset_path ,'rb') as f:
            self.path_dict = pickle.load(f)
            self.batch_size = batch_size
            self.win_size = win_size
            self.num_workers = num_workers
            self.kwargs = kwargs
            self.transform_train = transforms.Compose([
                GroupRandomRescale(0.8, key=['video', 'mask']),
                GroupRotate(15, key=['video', 'mask']),
                GroupCentralCrop((224,224), key=['video', 'mask']),
                Normalize(), 
                ToTensor()
                ])
            self.transform_val = transforms.Compose([
                GroupCentralCrop((224,224), key=['video', 'mask']),
                Normalize(),
                ToTensor()
                ])
            self._split_data()
            self._get_loder()

    
    def _get_loder(self):
        self.train_dataset = RgDataset(data_list=self.train_list, win_size=self.win_size, transform=self.transform_train, mode='train', **self.kwargs)
        self.val_dataset = RgDataset(data_list=self.val_list, win_size=self.win_size, transform=self.transform_val, mode='val', **self.kwargs)
        self.test_dataset = RgDataset(data_list=self.test_list, win_size=self.win_size, transform=self.transform_val, mode='test', **self.kwargs)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, drop_last=False)
    
    def _split_data(self):
        self.train_list = self.path_dict['train']
        self.val_list = self.path_dict['val']
        self.test_list = self.path_dict['test']


