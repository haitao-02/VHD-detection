import argparse
import yaml
parser = argparse.ArgumentParser(description='parse train config')
parser.add_argument('-c', '--config', required=True)
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
for c in config:
    print(c, config[c])
cuda_device = config['cuda_device']
config_file = 'mdoel_config.yaml'
cfg = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

import os
import random
import numpy as np
import glob
import torch
import torchvision.transforms as transforms
from .model import get_net
from .dataset import CWPoint_dataset
import pickle
import tqdm
from .functions import *
os.environ["CUDA_VISIBLE_DEVICES"] =cuda_device


# hyper-params
learning_rate = config['learning_rate']
batch_size = config['batch_size']
total_epoch = config['total_epoch']
steps = config['steps']
seed = config['seed']

num_workers = config['num_workers']
weight_decay = config['weight_decay']

train_path = config['train_path'] 
valid_path = config['valid_path']

train_tag = config['train_tag']

description = 'CWPoint_{}'.format(train_tag)
print('description： {}'.format(description))
save_dir = 'checkpoints/'+ description

#define
fix_randomness(seed)
best_acc_path = ''
best_loss_path = ''
best_acc = 0
best_acc_epoch = -1
best_loss_epoch = -1
best_loss = 100

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = CWPoint_dataset(train_path, transform)
valid_dataset = CWPoint_dataset(valid_path, transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            drop_last=False)
print('data size: training {}, validation {}.'.format(len(train_dataset), len(valid_dataset)))

model = get_pose_net(cfg, False)
model = torch.nn.DataParallel(model)
model = model.cuda()
criterion = JointsMSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(),
                            learning_rate, 
                            weight_decay=weight_decay)

for epoch in range(total_epoch):
    adjust_learning_rate(learning_rate, optimizer, weight_decay, epoch, steps)

    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
    valid_loss, valid_acc = validate(valid_loader, model, criterion, epoch)
    
    if epoch == 0:
        best_loss = valid_loss
        best_acc = valid_acc
        best_train_loss = train_loss
        best_acc_epoch = epoch
        best_loss_epoch = epoch
        save_checkpoint(model, epoch, prefix=save_dir+'/valid')
    else:
        if valid_loss <= best_loss:
            print('loss decreased from {best:.3f} to {loss:.3f} from epoch {epoch_num}'.format(best=best_loss, loss=valid_loss, epoch_num=best_loss_epoch))
            best_loss = valid_loss
            best_loss_epoch = epoch
            best_loss_path = save_checkpoint(model, epoch, prefix=save_dir+'/valid')
        else:
            print('loss did not decrease from {best:.3f} from epoch {epoch_num}'.format(best=best_loss, epoch_num=best_loss_epoch))
        if valid_acc >= best_acc:
            print('accuracy increased from {best:.3f} to {acc:.3f} from epoch {epoch_num}'.format(best=best_acc, acc=valid_acc, epoch_num=best_acc_epoch))
            best_acc = valid_acc
            best_acc_epoch = epoch
            best_acc_path = save_checkpoint(model, epoch, prefix=save_dir+'/valid')
        else:
            print('accuracy did not increase from {best:.3f} from epoch {epoch_num}'.format(best=best_acc, epoch_num=best_acc_epoch))
            if train_loss < best_train_loss:
                print('train loss decrease from {best:.3f} to {acc:.3f}'.format(best=best_train_loss, acc=train_loss))
                best_train_model_path = save_checkpoint(model, epoch, prefix = save_dir+'/train')
                best_train_loss = train_loss

#best acc
print('Training complete. Now testing best acc')
model.load_state_dict(torch.load(best_acc_path))
print('model loaded from checkpoint:', best_acc_path)
v_loss, v_acc = validate(valid_loader, model, criterion, -1)

#best loss
print('Training complete. Now testing best loss')
model.load_state_dict(torch.load(best_loss_path))
print('model loaded from checkpoint:', best_loss_path)
v_loss, v_acc = validate(valid_loader, model, criterion, -1)
