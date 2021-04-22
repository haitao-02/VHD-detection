import torch
import torch.nn as nn

def multi_class_dice_sqr_coef(y_pred, y_true, dims=(-2,-1), smooth=1e-6):
    intersection = torch.sum(y_true * y_pred, dim=dims)
    union = torch.sum(y_true * y_true, dim=dims) + torch.sum(y_pred * y_pred, dim=dims)
    raw = (2. * intersection + smooth) / (union + smooth)
    return (raw)

def multi_class_dice_sqr_loss(y_pred, y_true, dims=(-2,-1), **kwargs):
    return (1 - multi_class_dice_sqr_coef(y_pred, y_true, dims, **kwargs))

class RgLoss(nn.Module):
    def __init__(self):
        super(RgLoss, self).__init__()
        self.seg_loss = multi_class_dice_sqr_loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.kf_loss = nn.MSELoss()
    def __call__(self, seg_pred, cls_pred, kf_pred, seg_true, cls_true, kf_true):
        seg_loss = self.seg_loss(seg_pred, seg_true)

        seg_loss[~kf_true] = seg_loss[~kf_true]*0

        seg_loss = torch.mean(seg_loss)

        cls_loss = self.cls_loss(cls_pred, cls_true)
        kf_true = kf_true.view(-1, ).float()
        kf_loss = self.kf_loss(kf_pred, kf_true)
        return(0.7*seg_loss+0.1*cls_loss+0.2*kf_loss)
