import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import tqdm


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

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False, target_weight=None):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.target_weight = target_weight

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(self.target_weight[idx]),
                    heatmap_gt.mul(self.target_weight[idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


def train(train_loader, model, criterion, optimizer, epoch):
    print('Training...')
    losses = AverageMeter()
    accuracys = AverageMeter()
    # switch to train mode
    model.train()
    tbar = tqdm.tqdm(train_loader)
    for i, (img, target) in enumerate(tbar):
        img = img.cuda()
        target = target.cuda()
        # compute output
        output = model(img)
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        _, acc, cnt, _ = accuracy(output.detach().cpu().numpy(), target.cpu().numpy())
        accuracys.update(acc, cnt)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.set_description('epoch: {}, step: {}, loss: {loss:.3f}, accuracy: {acc:.3f}'.format(epoch, i, loss=losses.avg, acc=accuracys.avg))
    # epoch end 
    print('epoch: {} end, loss: {loss:.3f}, accuracy: {acc:.3f}'.format(epoch, loss=losses.avg, acc=accuracys.avg))
    return losses.avg, accuracys.avg

def validate(valid_loader, model, criterion, epoch):
    print('Evaluating...')
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        accuracys = AverageMeter()
        for i, (img, target) in enumerate(valid_loader):
            img = img.cuda()
            target = target.cuda()
            # compute output
            output = model(img)
            loss = criterion(output, target)
            losses.update(loss.item(), img.size(0))
            _, acc, cnt, _ = accuracy(output.detach().cpu().numpy(), target.cpu().numpy())
            accuracys.update(acc, cnt)
        print('epoch: {}, loss: {loss:.3f}, accuracy: {acc:.3f}'.format(epoch, loss=losses.avg, acc=accuracys.avg))
    return losses.avg, accuracys.avg