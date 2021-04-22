import os
import random
import numpy as np
import torch
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

def cal_weighted_acc(conf, weight=None):
    if weight is None:
        ws = [1/conf.shape[0]]*conf.shape[0]
    else:
        ws = [w/sum(weight) for w in weight]
    true_sum = np.sum(conf, axis=1)
    true_pre_num = [conf[i, i] for i in range(conf.shape[0])]
    acc = 0
    for tpn, ts, w in zip(true_pre_num, true_sum, ws):
        if ts==0:
            continue
        acc += (tpn/ts)*w
    return acc*100

def train(train_loader, model, criterion, num_classes, optimizer, epoch):
    print('Training...')
    losses = AverageMeter()
    conf = ConfusionMatrix(num_classes)
    # switch to train mode
    model.train()
    tbar = tqdm.tqdm(train_loader)
    for i, (x1, x2, y) in enumerate(tbar):
        x1 = x1.cuda()
        x2 = x2.cuda()
        y = y.cuda()
        # compute output
        output = model(x1, x2)
        loss = criterion(output, y)
        losses.update(loss.item(), x1.size(0))
        conf.update(output.data, y)
        acc = cal_weighted_acc(conf.mat)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.set_description('epoch: {}, step: {}, loss: {loss:.3f}, accuracy: {acc:.3f}'.format(epoch, i, loss=losses.avg, acc=acc))
    # epoch end 
    epoch_acc = cal_weighted_acc(conf.mat)
    print('epoch: {}, epoch end, loss: {loss:.3f}, accuracy: {acc:.3f}'.format(epoch, loss=losses.avg, acc=epoch_acc))
    print('confusion matrix: ')
    print(conf.mat)
    return losses.avg, epoch_acc

def validate(valid_loader, model, criterion, num_classes, epoch):
    print('Evaluating...')
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        conf = ConfusionMatrix(num_classes)
        tbar = tqdm.tqdm(valid_loader)
        for i, (x1, x2, y) in enumerate(tbar):
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()
            # compute output
            output = model(x1, x2)
            loss = criterion(output, y)
            losses.update(loss.item(), x1.size(0))
            conf.update(output.data, y)
        epoch_acc = cal_weighted_acc(conf.mat)
        print('epoch: {}, loss: {loss:.3f}, accuracy: {acc:.3f}'.format(epoch, loss=losses.avg, acc=epoch_acc))
        print('confusion matrix: ')
        print(conf.mat)
    return losses.avg, epoch_acc, conf.mat