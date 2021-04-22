import random
import numbers
import numpy as np
import math
from skimage.transform import resize, rotate, rescale

class GroupResize:
    def __init__(self, shape, key=['video']):
        if isinstance(shape, numbers.Number):
            self.shape = (int(shape), int(shape))
        else:
            self.shape = shape
        self.key = key

    def __call__(self, sample):
        for k in self.key:
            order = 0 if k in ['mask'] else 1
            img_group = sample[k]
            if img_group.ndim == 4:
                tt, h, w, tc = img_group.shape
                th, tw = self.shape[:2]
                target_shape = (tt, th, tw, tc)
            elif img_group.ndim == 3:
                tt, h, w = img_group.shape
                th, tw = self.shape
                target_shape = (tt, th, tw)
            else:
                raise ValueError('The dimension of {:s} should be either 3 or 4'.format(k))
            scale = [th/h, tw/w]
            if w == tw and h == th:
                out_images = img_group
            else:
                out_images = resize(img_group, target_shape, order=order, preserve_range=True)
            sample[k] = out_images
            if k in ['mask']:
                sample[k] = sample[k].astype('uint8')
        return sample

class GroupRotate:
    def __init__(self, angle, key=['video'], resize=False):
        self.angle = angle
        self.key = key
        self.resize = resize
    def __call__(self, sample):
        alpha = random.randint(-self.angle, self.angle)
        for k in self.key:
            order = 0 if k in ['mask'] else 1
            img_group = sample[k]
            out_images = []
            if self.angle == 0:
                out_images = img_group
            else:
                for img in img_group:
                    out_images.append(rotate(img, alpha, preserve_range=True, resize=self.resize, order=order))
            sample[k] = np.asarray(out_images)
            if k in ['mask']:
                sample[k] = sample[k].astype('uint8')
            if k in ['vf', 'flow']:
                sample[k][...,0] = sample[k][...,0]*math.cos(alpha/180*math.pi) + sample[k][...,1]*math.sin(alpha/180*math.pi)
                sample[k][...,1] = -sample[k][...,0]*math.sin(alpha/180*math.pi) + sample[k][...,1]*math.cos(alpha/180*math.pi)
        return sample

class GroupRandomRescale:
    def __init__(self, scale, key=['video']):
        self.scale = scale
        self.key = key
    def __call__(self, sample):
        s = random.uniform(self.scale, 1)
        for k in self.key:
            order = 0 if k in ['mask'] else 1
            img_group = sample[k]
            if img_group.ndim == 3:
                out_images = rescale(img_group, (1,s,s), order=order, preserve_range=True, multichannel=False)
            elif img_group.ndim == 4:
                out_images = rescale(img_group, (1,s,s), order=order, preserve_range=True, multichannel=True)
            else:
                raise ValueError('The dimension of {:s} should be either 3 or 4'.format(k))
            sample[k] = out_images
            if k in ['mask']:
                sample[k] = sample[k].astype('uint8')
        return sample

class GroupCentralCrop:
    def __init__(self, target_size, key=['video']):
        self.target_size = target_size
        self.key = key
    def __call__(self, sample):
        th, tw = self.target_size
        for k in self.key:
            h, w = sample[k].shape[1:3]
            # crop
            if th < h:
                sh, eh = (h-th) // 2, th + (h-th) // 2
            else:
                sh, eh = 0, h
            if tw < w:
                sw, ew = (w-tw) // 2, tw + (w-tw) // 2
            else:
                sw, ew = 0, w
            sample[k] = sample[k][:,sh:eh, sw:ew]
            # padding
            h, w = eh-sh, ew-sw
            dh, dw = (th -h) // 2, (tw-w) // 2
            if sample[k].ndim == 4:
                sample[k] = np.pad(sample[k], ((0,0), (dh, th-h-dh), (dw, tw-w-dw), (0,0)), mode='constant')
            elif sample[k].ndim == 3:
                sample[k] = np.pad(sample[k], ((0,0), (dh, th-h-dh), (dw, tw-w-dw)), mode='constant')
            else:
                raise ValueError('The dimension of {:s} should be either 3 or 4'.format(k))
        return(sample)

