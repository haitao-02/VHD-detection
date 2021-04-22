import os
import torch
import torchvision
import numpy as np
import random
from PIL import Image
from .VideoClsModel import TwoView_Classification_Model
from .transforms import *
from .utils import *

class ASClsInfer:
    """
    input: folder_path, which has PLAX-2D and PSAX-A-2D npy file
    output: dict -pred_label: 0-negtive, 1-positive
                -prob: 0~1, probability of predicted positive
    """
    def __init__(self, model_path=None, use_gpu=True, return_output=False):
        self.model_path = model_path
        self.transform = torchvision.transforms.Compose(
            [VideoArrayToPIL(convert_gray=True),
            Stack(),
            ToTorchFormatTensor(div=True)]
        )
        self.use_gpu = use_gpu
        self.return_output = return_output
    
    def load_model(self):
        self.model = TwoView_Classification_Model(class_num=2, add_bias=True)
        if self.model_path is not None:
            self.model.load_state_dict(torch.load(self.model_path))
        if self.use_gpu:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.model.eval()
    
    def __call__(self, folder):
        xs = self.preprocess(folder)
        xs = [torch.unsqueeze(v, 0) for v in xs]
        if self.use_gpu:
            xs = [x.cuda() for x in xs]
        with torch.no_grad():
            output = self.model(*xs)
            pred = output.argmax().item()
            prob = torch.nn.functional.softmax(output, dim=1)
            poss = prob[0][1].item()
        res = {
            'pred_label': pred,
            'prob': poss
        }
        if self.return_output:
            return output
        else:
            return res

    
    def preprocess(self, folder_path):
        plax = get_view(folder_path, 'PLAX-2D')
        psax = get_view(folder_path, 'PSAX-A-2D')
        if len(plax)==0 and len(psax)==0:
            raise Exception('no available view found in {}'.format(folder_path))
        views = []
        for v in [plax, psax]:
            if len(v)==0:
                array = np.zeros((64, 224, 224, 3)).astype('uint8')
                array = self.transform(array)
                views.append(array)
            else:
                array = np.load(v)
                array = remove_info(array)
                array = clip_video(array, size=64, random_clip=False)
                array = crop_resize_video(array, size=224)
                array = self.transform(array)
                views.append(array)
        return views

def get_view(folder, view):
    files = os.listdir(folder)
    out = ''
    for f in files:
        if view+'.npy' in f:   #npy file
            out = os.path.join(folder, f)
            break
    return out

if __name__=='__main__':
    infer = ASClsInfer(use_gpu=False, return_output=True)
    infer.load_model()
    test_output = infer('example')
    print(test_output.shape)
    print(test_output)
