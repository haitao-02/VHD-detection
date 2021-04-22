import numpy as np
import torch
import torchvision.transforms as transforms
from .model import get_net
import yaml
from .functions import get_max_preds
import cv2
import os
current_dir = os.path.split(__file__)[0]


class CWPoints_infer:
    def __init__(self, model_path=None, model_config=os.path.join(current_dir, 'model_config.yaml'), use_gpu=True):
        self.model_path = model_path
        self.model_config = model_config
        self.transform = transforms.Compose([
            transforms.ToTensor()
            ])
        self.use_gpu = use_gpu
    
    def load_model(self):
        cfg = yaml.load(open(self.model_config, 'r'), Loader=yaml.FullLoader)
        self.model = get_net(cfg, False)
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path))
        if self.use_gpu:
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        self.model.eval()
    
    def __call__(self, img):  #img shape: [box_size, box_size, 3]
        ipt = cv2.resize(img, (256,256))
        with torch.no_grad():
            ipt = self.transform(ipt)
            if self.use_gpu:
                ipt = ipt.cuda()
            out = self.model(ipt.unsqueeze(0))  #[1,2,64,64]
            pre, _ = get_max_preds(out.detach().cpu().numpy())
        points = pre[0]
        reloc = points/64
        return reloc 

if __name__ == '__main__':
    infer = CWPoints_infer(use_gpu=False)
    infer.load_model()
    img = np.random.randint(0, 256, (88,88,3)).astype('uint8')
    loc = infer(img)
    print(loc)