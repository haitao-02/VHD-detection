import argparse
import yaml
parser = argparse.ArgumentParser(description='parse train config')
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
for c in config:
    print(c, config[c])
cuda_device = config['cuda_device']

import os
import random
import numpy as np
import torch
import torchvision
from .model import UNet
from .dataset import CWDataset
from .functions import *
import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] =cuda_device

# hyper-params
num_classes = config['num_classes']
batch_size = config['batch_size']
num_workers = config['num_workers']
learning_rate = config['learning_rate']
momentum = config['momentum']
weight_decay = config['weight_decay']

total_epoch = config['total_epoch']
steps = config['steps']
seed = config['seed']
class_weight = config['class_weight']

train_tag = config['train_tag']

description = 'AS_CW_{}'.format(train_tag)
print('description：{}'.format(description))
save_dir = 'checkpoints/' + description

class_weight = np.array(class_weight)
print('class weight: ', class_weight)


fix_randomness(seed)

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
valid_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_path = config['train_path'] 
valid_path = config['valid_path']
data_root = config['data_root']

train_dataset = CWDataset(train_path, data_root, train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=num_workers, 
                                           pin_memory=True,
                                           drop_last=True)


valid_dataset = CWDataset(valid_path, data_root, valid_transform)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                        batch_size=batch_size, 
                                        shuffle=False,
                                        num_workers=num_workers, 
                                        pin_memory=True,
                                        drop_last=False)

print('data size: training {}, validation {}.'.format(len(train_dataset), len(valid_dataset)))

model = UNet(num_classes= num_classes)
model = torch.nn.DataParallel(model)
model = model.cuda()
criterion = MulticlassDiceLoss(weight=class_weight).cuda()
optimizer = torch.optim.Adam(model.parameters(),
                            learning_rate, 
                            weight_decay=weight_decay)

for epoch in range(total_epoch):
    adjust_learning_rate(learning_rate, optimizer, weight_decay, epoch, steps)

    train_loss, train_dice, train_dice_cls = train(train_loader, model, criterion, optimizer, epoch)
    valid_loss, valid_dice, valid_dice_cls = validate(valid_loader, model, criterion, epoch)
    
    if epoch == 0:
        best_loss = valid_loss
        best_dice = valid_dice
        best_loss_epoch = epoch
        best_dice_epoch = epoch
        save_checkpoint(model, epoch, prefix=save_dir+'/valid')
    else:
        if valid_loss <= best_loss:
            print('loss decreased from {best:.3f} to {loss:.3f} from epoch {epoch_num}'.format(best=best_loss, loss=valid_loss, epoch_num=best_loss_epoch))
            best_loss = valid_loss
            best_loss_epoch = epoch
            best_loss_path = save_checkpoint(model, epoch, prefix=save_dir+'/valid')
        else:
            print('loss did not decrease from {best:.3f} from epoch {epoch_num}'.format(best=best_loss, epoch_num=best_loss_epoch))
        if valid_dice >= best_dice:
            print('dice increased from {best:.3f} to {acc:.3f} from epoch {epoch_num}'.format(best=best_dice, acc=valid_dice, epoch_num=best_dice_epoch))
            best_dice = valid_dice
            best_dice_epoch = epoch
            best_dice_path = save_checkpoint(model, epoch, prefix=save_dir+'/valid')
        else:
            print('dice did not increase from {best:.3f} from epoch {epoch_num}'.format(best=best_dice, epoch_num=best_dice_epoch))
        

#best acc
print('Training complete. Now testing best dice')
model.load_state_dict(torch.load(best_dice_path))
print('model loaded from checkpoint:', best_dice_path)
v_loss, v_dice, v_dice_cls = validate(valid_loader, model, criterion, -1)
#best loss
print('Training complete. Now testing best loss')
model.load_state_dict(torch.load(best_loss_path))
print('model loaded from checkpoint:', best_loss_path)
v_loss, v_dice, v_dice_cls = validate(valid_loader, model, criterion, -1)