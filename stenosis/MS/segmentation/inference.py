import torch
import torchvision
import numpy as np
from .model import UNet
from PIL import Image
import cv2
from .utils import *

class MSCWInfer:
    def __init__(self, model_path=None, use_gpu=True):
        self.model_path = model_path
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])
        self.use_gpu = use_gpu

    def load_model(self):
        self.model = UNet(num_classes=2)
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path))
        if self.use_gpu:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.model.eval()
    
    def __call__(self, array):
        ipt_img, pad_dim, diff = self.preprocess(array)
        with torch.no_grad():
            ipt = self.transform(ipt_img)
            if self.use_gpu:
                ipt = ipt.cuda()
            output = self.model(torch.unsqueeze(ipt, 0))
            prob = output.argmax(dim=1)
            pre = prob.cpu().numpy()
        mask = pre[0].astype('uint8')
        mask[np.where(mask>0)]=255  #(512,512)
        opt_mask = self.postprocess(mask, array.shape, pad_dim, diff)
        return opt_mask

        
    def preprocess(self, array):
        ori_img = array
        h, w = ori_img.shape[0], ori_img.shape[1]
        diff = abs(h-w)//2
        if h<w:
            pad_dim = 'h'
        else:
            pad_dim = 'w'
        ipt_img = crop_resize_image(ori_img, 512)
        return ipt_img, pad_dim, diff
    
    def postprocess(self, mask, ori_size, pad_dim, diff):
        if pad_dim=='h':
            mask = resize_image(mask, size=ori_size[1], convert_RGB=False)
            opt_mask = mask[diff:-diff, :]
        else:
            mask = resize_image(mask, size=ori_size[0], convert_RGB=False)
            opt_mask = mask[:, diff:-diff]
        opt_mask = np.where(opt_mask>127, 255, 0).astype('uint8')
        return opt_mask




if __name__ == '__main__':
    img = np.random.rand(0, 256, (600,800,3)).astype('uint8')
    infer = MSCWInfer(use_gpu=False)
    infer.load_model()
    mask = infer(img)
    print('mask shape:', mask.shape)
