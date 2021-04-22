import torch
import numpy as np
import math
from convertions import *
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc

class MetricEvaluator:
    def __init__(self, dict):
        self.metrics = dict
        self.rlt = {}

    def reset(self):
        self.rlt = {}
        for k in self.metrics:
            self.metrics[k].reset()
    
    def update(self, *args):
        for k in self.metrics:
            self.metrics[k].update(*args)
    
    def eval(self):
        for k in self.metrics:
            self.rlt[k] = self.metrics[k].eval()
        return(self.rlt)

def rlt_table(rlt, title='Result', field_names=['Metric', 'Value'], tb=None):
    if tb is None:
        tb = PrettyTable(title=title, field_names=field_names)
        tb.title = title
    for k in rlt:
        tb.add_row([k, rlt[k]])
    return(tb)

def multi_class_iou(y_pred, y_true, cls_idx=None, dims= (-2,-1), smooth=1e-6):
    one_hot = prob_to_one_hot(y_pred, cuda=True).float()
    cls_num = one_hot.size()[1]
    one_hot_label = y_true
    intersection = torch.sum(one_hot_label * one_hot, dim=dims)
    union = torch.sum(one_hot_label, dim=dims) + torch.sum(one_hot, dim=dims) - intersection
    # iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    iou = (intersection + smooth) / (union + smooth)
    if cls_idx is not None:
        return iou[...,cls_idx].item(), len(iou)
    else:
        return iou.data.cpu().numpy(), len(iou)

class MultiClassIoU:
    def __init__(self, dims=(-2,-1), cls_idx=None, smooth=1e-6):
        self.dims = dims
        self.smooth = smooth
        self.cls_idx = cls_idx
    def __call__(self, y_pred, y_true):
        return(multi_class_iou(y_pred, y_true, cls_idx=self.cls_idx, dims=self.dims, smooth=self.smooth))

def multi_label_iou(y_pred, y_true, cls_idx=None, dims= (-2,-1), smooth=1e-6):
    one_hot_pred = (y_pred > 0.5).float()
    one_hot_label = y_true
    intersection = torch.sum(one_hot_label * one_hot_pred, dim=dims)
    union = torch.sum(one_hot_label, dim=dims) + torch.sum(one_hot_pred, dim=dims) - intersection
    # iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    iou = (intersection + smooth) / (union + smooth)
    if cls_idx is not None:
        return iou[cls_idx].item(), len(iou)
    else:
        return iou.data.cpu().numpy(), len(iou)

class MultiLabelIoU:
    def __init__(self, dims=(-2,-1), cls_idx=None, smooth=1e-6):
        self.dims = dims
        self.smooth = smooth
        self.cls_idx = cls_idx
    def __call__(self, y_pred, y_true):
        return(multi_label_iou(y_pred, y_true, cls_idx=self.cls_idx, dims=self.dims, smooth=self.smooth))

def multi_class_dice(y_pred, y_true, cls_idx=None, dims= (-2,-1), smooth=1e-6):
    one_hot = prob_to_one_hot(y_pred, cuda=True).float()
    cls_num = one_hot.size()[1]
    one_hot_label = y_true
    intersection = 2.0 * torch.sum(one_hot_label * one_hot, dim=dims)
    union = torch.sum(one_hot_label, dim=dims) + torch.sum(one_hot, dim=dims)
    # dice = torch.mean((intersection + smooth) / (union + smooth), dim=0)
    dice = (intersection + smooth) / (union + smooth)
    if cls_idx is not None:
        return dice[cls_idx].item(), len(dice)
    else:
        return dice.data.cpu().numpy(), len(dice)

class MultiClassDice:
    def __init__(self, dims=(-2,-1), cls_idx=None, smooth=1e-6):
        self.dims = dims
        self.smooth = smooth
        self.cls_idx = cls_idx
    def __call__(self, y_pred, y_true):
        return(multi_class_dice(y_pred, y_true, cls_idx=self.cls_idx, dims=self.dims, smooth=self.smooth))

def multi_label_recall(y_pred, y_true, threshold=0.5):
    y_pred[y_pred>threshold] = 1
    y_pred[y_pred<=threshold] = 0
    # tp = torch.sum(y_pred*y_true, dim=0).data.cpu().numpy()
    tp = (y_pred*y_true).data.cpu().numpy()
    gp = torch.sum(y_true, dim=0).data.cpu().numpy()
    # tp = np.divide(tp, gp, out=np.zeros_like(tp), where=gp!=0)
    return(tp, gp)

class MultiLabelRecall:
    def __init__(self):
        pass
    def __call__(self, y_pred, y_true):
        num_classes = y_pred.size()[-1]
        y_pred = y_pred.view(-1, num_classes)
        y_true = y_true.view(-1, num_classes)
        return(multi_label_recall(y_pred, y_true))

def multi_label_precision(y_pred, y_true, threshold=0.5):
    y_pred[y_pred>threshold] = 1
    y_pred[y_pred<=threshold] = 0
    # tp = torch.sum(y_pred*y_true, dim=0).data.cpu().numpy()
    tp = (y_pred*y_true).data.cpu().numpy()
    pp = torch.sum(y_pred, dim=0).data.cpu().numpy()
    # tp = np.divide(tp, pp, out=np.zeros_like(tp), where=pp!=0)
    return(tp, pp)

class MultiLabelPrecision:
    def __init__(self):
        pass
    def __call__(self, y_pred, y_true):
        num_classes = y_pred.size()[-1]
        y_pred = y_pred.view(-1, num_classes)
        y_true = y_true.view(-1, num_classes)
        return(multi_label_precision(y_pred, y_true))

def multi_label_accuracy(y_pred, y_true, threshold=0.5):
    y_pred[y_pred>threshold] = 1
    y_pred[y_pred<=threshold] = 0
    # correct = torch.sum((torch.norm(y_pred - y_true, dim=-1, p=1)==0).view(-1,), dim=0, keepdim=True).data.cpu().numpy() / len(y_pred)
    correct = (torch.norm(y_pred - y_true, dim=-1, p=1)==0).view(-1,).float().data.cpu().numpy()
    return(correct, len(y_pred))

class MultiLabelAccuracy:
    def __init__(self):
        pass
    def __call__(self, y_pred, y_true):
        num_classes = y_pred.size()[-1]
        y_pred = y_pred.view(-1, num_classes)
        y_true = y_true.view(-1, num_classes)
        return(multi_label_accuracy(y_pred, y_true))

class Accuracy:
    def __init__(self,dim=1, one_hot=False):
        self.dim = dim
        self.one_hot = one_hot
    def __call__(self, y_pred, y_true):
        num_classes = y_pred.size()[-1]
        if self.one_hot:
            y_true = y_true.view(-1, )
            y_true = tensor_label_to_one_hot(y_true, num_classes)
        else:
            y_true = y_true.view(-1, num_classes)
        y_pred = tensor_prob_to_one_hot(y_pred, dim=self.dim).view(-1, num_classes)
        return(multi_label_accuracy(y_pred, y_true))

class Recall:
    def __init__(self, dim = 1, one_hot=False):
        self.dim = dim
        self.one_hot = one_hot
    def __call__(self, y_pred, y_true):
        num_classes = y_pred.size()[-1]
        if self.one_hot:
            y_true = y_true.view(-1, )
            y_true = tensor_label_to_one_hot(y_true, num_classes)
        else:
            y_true = y_true.view(-1, num_classes)
        y_pred = tensor_prob_to_one_hot(y_pred, dim=self.dim).view(-1, num_classes)
        return(multi_label_recall(y_pred, y_true))

class Precision:
    def __init__(self, dim=1, one_hot=False):
        self.dim = dim
        self.one_hot = one_hot
    def __call__(self, y_pred, y_true):
        num_classes = y_pred.size()[-1]
        if self.one_hot:
            y_true = y_true.view(-1, )
            y_true = tensor_label_to_one_hot(y_true, num_classes).view(-1, num_classes)
        else:
            y_true = y_true.view(-1, num_classes)
        y_pred = tensor_prob_to_one_hot(y_pred, dim=self.dim).view(-1, num_classes)
        return(multi_label_precision(y_pred, y_true))
        

def roc_auc(y_pred, y_true):
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

class Auc:
    def __init__(self, one_hot_class_num=0):
        self.one_hot_class_num = one_hot_class_num
    def __call__(self, y_pred, y_true):
        if self.one_hot_class_num > 0:
            y_true = tensor_label_to_one_hot(y_true, self.one_hot_class_num)
        return(roc_auc(y_pred, y_true))

class MetricWrapper:
    def __init__(self, metric, idx=0):
        self.idx = idx
        self.prediction = []
        self.ground_truth = []
        self.output = 0
        self.metric = metric
    
    def reset(self):
        self.__init__(self.metric, self.idx)

    def update(self, y_pred, y_true):
        self.prediction.append(y_pred.data.cpu().numpy()[:,self.idx])
        self.ground_truth.append(y_true.data.cpu().numpy()[:, self.idx])
    
    def eval(self):
        self.prediction = np.concatenate(self.prediction, axis=0)
        self.ground_truth = np.concatenate(self.ground_truth, axis=0)
        self.output = self.metric(self.prediction, self.ground_truth)
        return self.output

class AvgMeterWrapper:
    def __init__(self, metric):
        self.metric = metric
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def reset(self):
        self.__init__(self.metric)

    def update(self, *args):
            value, sample_num = self.metric(*args)
            self.val = value
            self.sum = self.sum  + np.sum(value, axis=0)
            self.count += sample_num

    def eval(self):
        self.avg = np.divide(self.sum, self.count, out=np.zeros_like(self.sum), where=self.count!=0)
        return(self.avg)

class ConfusionMatrix:
    def __init__(self, num_classes, dim=1):
        self.dim = dim
        self.num_classes = num_classes
        self.m = np.zeros((num_classes, num_classes), dtype='int32')
    def reset(self):
        self.__init__(self.num_classes, self.dim)
    def update(self, y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=self.dim).data.cpu().numpy()
        y_true = torch.argmax(y_true, dim=self.dim).data.cpu().numpy()
        for p, g in zip(y_pred, y_true):
            self.m[g, p] += 1
    def eval(self):
        return self.m



