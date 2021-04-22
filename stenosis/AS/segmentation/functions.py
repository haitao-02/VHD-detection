import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm


class ConfusionMatrix(object):
    def __init__(self, class_num):
        self.mat = np.zeros((class_num, class_num))

    def reset(self):
        self.mat = np.zeros((class_num, class_num))

    def update(self, output, y):
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t().view(-1)
        y = y.view(-1)
        for i in range(y.size(0)):
            self.mat[int(y[i]), int(pred[i])] += 1  

def fix_randomness(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def adjust_learning_rate(initial_lr, optimizer, weight_decay, epoch, steps):
    """Sets the learning rate to the initial LR decayed by 3 every stage"""
    power = sum([epoch>=step for step in steps])
    multiplier = 0.3 ** power
    lr = initial_lr * multiplier
    print('current learning rate', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        param_group['weight_decay'] = weight_decay

def save_checkpoint(model, epoch, prefix='./checkpoints'):
    filename = os.path.join(prefix, 'epoch_'+ str(epoch) + '.pth')
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    torch.save(model.state_dict(), filename)
    print('saved checkpoint to {}'.format(filename))
    return filename

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch):
    print('Training...')
    losses = AverageMeter()
    dices = AverageMeter()
    each_cls_dices = AverageMeter()
    # switch to train mode
    model.train()
    tbar = tqdm.tqdm(train_loader)
    for i, (x, y) in enumerate(tbar):
        x = x.cuda()
        y = y.cuda()
        # compute output
        output = model(x)
        loss = criterion(output, y)
        losses.update(loss.item(), x.size(0))
        each_cls_dice, dice = dice_coeff(output, y)
        each_cls_dices.update(each_cls_dice.detach().cpu().numpy(), x.size(0))
        dices.update(dice.item(), x.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.set_description('epoch: {}, step: {}, loss: {loss:.3f}, dice: {coe:.3f}'.format(epoch, i, loss=losses.avg, coe=dices.avg))
    # epoch end    
    print('epoch: {}, epoch end, loss: {loss:.3f}, dice: {coe:.3f}'.format(epoch, loss=losses.avg, coe=dices.avg))
    return losses.avg, dices.avg, each_cls_dices.avg

def validate(valid_loader, model, criterion, epoch):
    print('Evaluating...')
    losses = AverageMeter()
    dices = AverageMeter()
    each_cls_dices = AverageMeter()
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(valid_loader):
            x = x.cuda()
            y = y.cuda()
            # compute output
            output = model(x)
            loss = criterion(output, y)
            losses.update(loss.item(), x.size(0))
            each_cls_dice, dice = dice_coeff(output, y)
            each_cls_dices.update(each_cls_dice.detach().cpu().numpy(), x.size(0))
            dices.update(dice.item(), x.size(0))

    print('epoch: {}, loss: {loss:.3f}, dice: {coe:.3f}'.format(epoch, loss=losses.avg, coe=dices.avg))
    
    return losses.avg, dices.avg, each_cls_dices.avg


class MulticlassDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(MulticlassDiceLoss, self).__init__()
        if weight is None:
            self.weight = torch.ones(num_cls, 1)
        else:
            self.weight = torch.tensor(weight.T, dtype = torch.float32)

    def forward(self, output, target, smooth=1): #output:[B, C, H, W] target: [B, H, W]
        num_cls = output.shape[1]
        #reshape
        one_hot_tar = label_to_one_hot(target, num_cls=num_cls, use_GPU=output.is_cuda)  #label [B, H, W] --> one_hot [B, C, H, W]
        output = output.softmax(dim=1) #output with cls_possibility of each pixel
        #calculate
        intersection = (output * one_hot_tar).sum(dim=(2,3))
        union = output.sum(dim=(2,3)) + one_hot_tar.sum(dim=(2,3))
        loss = 1- (2*intersection + smooth) / (union + smooth) #(B,C)
        
        weight = self.weight.cuda()
        dice_loss = torch.mm(loss.mean(dim = 0, keepdim=True), weight) / num_cls
        #dice_loss = torch.mm(torch.mean(loss, dim = 0, keepdim=True), weight) / num_cls
        return dice_loss

def dice_coeff(output, target, smooth=1):
    num_cls = output.shape[1]
    one_hot_tar = label_to_one_hot(target, num_cls=num_cls, use_GPU=output.is_cuda)
    output = output.softmax(dim=1)
    one_hot_out = prob_to_one_hot(output, use_GPU=output.is_cuda)
    intersection = (one_hot_out * one_hot_tar).sum(dim=(2,3))
    union = one_hot_out.sum(dim=(2,3)) + one_hot_tar.sum(dim=(2,3))
    dice_coeff = (2*intersection + smooth) / (union + smooth) #(B,C)
    return dice_coeff.mean(dim=0) , dice_coeff.mean()

def label_to_one_hot(label, num_cls, use_GPU):
    one_hot = torch.zeros(label.shape[0], num_cls, label.shape[1], label.shape[2])
    label = torch.unsqueeze(label, dim=1)
    if use_GPU:
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, label, 1.0)
    return one_hot

def prob_to_one_hot(prob, use_GPU):
    one_hot = torch.zeros(prob.shape)
    one_hot.requires_grad_(True)
    argmx_prob = torch.argmax(prob, dim=1, keepdim=True)
    if use_GPU:
        one_hot = one_hot.cuda()
    one_hot.scatter_(1, argmx_prob, 1)
    return one_hot