import numpy as np
import cv2
from PIL import Image
import PIL
import random


def remove_info(array): #remove info on the image
    video = array.copy()
    l = video.shape[0] //2
    v1 = video[:l]
    v2 = video[-l:]
    mask = np.sum(v1-v2, axis=0)
    video[:,mask==0] = 0
    return video

def convertRGB(array):
    shape = array.shape
    assert len(shape) in [2,3]
    if len(shape) == 2: # image
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        array = np.array([cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) for a in array])
    return array

def crop_resize_video(array, size=256, convert_RGB=True):
    imgs = [make_img_square(a) for a in array]
    video = np.array([resize_image(a, size, convert_RGB) for a in imgs])
    return video

def make_img_square(array):
    if array.shape[0] != array.shape[1]:
        diff = abs(array.shape[1] - array.shape[0]) // 2
        if len(array.shape) == 3: 
            if array.shape[0] < array.shape[1]:
                array = np.pad(array, ((diff,diff), (0,0), (0,0)), mode='constant')
            else:
                array = np.pad(array, ((0,0), (diff,diff), (0,0)), mode='constant')
        else:
            if array.shape[0] < array.shape[1]:
                array = np.pad(array, ((diff,diff), (0,0)), mode='constant')
            else:
                array = np.pad(array, ((0,0), (diff,diff)), mode='constant')
    return array

def resize_image(array, size, convert_RGB=True):
    img = Image.fromarray(array.astype('uint8'))
    if convert_RGB:
        img = img.convert('RGB')
    img = img.resize((size, size), resample=PIL.Image.BILINEAR)
    return np.array(img)


def clip_video(video, size = 64, random_clip=False): #clip video to size(64) padding itself
    video0 = [v for v in video]
    video = [v for v in video]
    while (len(video) < size):
        video += video0
    try:
        start = random.choice(range(len(video) - size))
    except:
        start = 0
    if not random_clip:
        start = 0
    video = video[start:start + size]
    video = np.array(video)
    return video
