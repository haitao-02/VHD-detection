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
import numpy as np
import torch
import torchvision
from .VideoClsModel import TwoView_Classification_Model
from .functions import *
from .dataset import TwoViewDataset
from .transforms import *
from .utils import *
os.environ["CUDA_VISIBLE_DEVICES"] =cuda_device

# hyper-params
num_classes = config['num_classes']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
total_epoch = config['total_epoch']
steps = config['steps']
seed = config['seed']
num_workers = config['num_workers']
momentum = config['momentum']
weight_decay = config['weight_decay']

view_dropout = config['view_dropout']
drop_rate = config['drop_rate']
add_bias= config['add_bias']
clip_size = config['clip_size']
train_tag = config['train_tag']

description = 'MS_2views_{}'.format(train_tag)
print('description： {}'.format(description))
save_dir = 'checkpoints/' + description

train_class_weight = config['train_class_weight']
print('train class weight', train_class_weight)
valid_class_weight = config['valid_class_weight']
print('valid class weight', valid_class_weight)


#define
fix_randomness(seed)
best_acc_path = ''
best_loss_path = ''
best_acc = 0
best_acc_epoch = -1
best_loss_epoch = -1
best_loss = 100

train_transform = torchvision.transforms.Compose([
    VideoArrayToPIL(convert_gray=True),
    GroupMultiScaleCrop(224,[1,0.9,0.9,0.9]),
    GroupRotation(),
    Stack(),
    ToTorchFormatTensor(div=True),
])
valid_transform = torchvision.transforms.Compose([
    VideoArrayToPIL(convert_gray=True),
    Stack(),
    ToTorchFormatTensor(div=True),
])

train_path = config['train_path'] 
valid_path = config['valid_path']

train_dataset = TwoViewDataset(train_path, train_transform, view_dropout=view_dropout, drop_rate=drop_rate, clip_size=clip_size, random_clip=True)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=num_workers, 
                                           pin_memory=True,
                                           drop_last=True)


valid_dataset = TwoViewDataset(valid_path, valid_transform, clip_size=clip_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                        batch_size=batch_size, 
                                        shuffle=False,
                                        num_workers=num_workers, 
                                        pin_memory=True,
                                        drop_last=False)

print('data size: training {}, validation {}.'.format(len(train_dataset), len(valid_dataset)))

model = TwoView_Classification_Model(class_num=num_classes, add_bias=add_bias)
model = torch.nn.DataParallel(model)
model = model.cuda()
train_criterion = torch.nn.CrossEntropyLoss(torch.Tensor(train_class_weight)).cuda()
valid_criterion = torch.nn.CrossEntropyLoss(torch.Tensor(valid_class_weight)).cuda()
optimizer = torch.optim.Adam(model.parameters(),
                            learning_rate, 
                            weight_decay=weight_decay)

for epoch in range(total_epoch):
    adjust_learning_rate(learning_rate, optimizer, weight_decay, epoch, steps)

    train_loss, train_acc = train(train_loader, model, train_criterion, num_classes, optimizer, epoch)
    valid_loss, valid_acc, conf = validate(valid_loader, model, valid_criterion, num_classes, epoch)
    
    if epoch == 0:
        best_loss = valid_loss
        best_acc = valid_acc
        best_train_acc = train_acc
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
            if train_acc > best_train_acc:
                print('train accuracy improved from {best:.3f} to {acc:.3f}'.format(best=best_train_acc, acc=train_acc))
                best_train_model_path = save_checkpoint(model, epoch, prefix = save_dir+'/train')
                best_train_acc = train_acc

#best acc
print('Training complete. Now testing best acc')
model.load_state_dict(torch.load(best_acc_path))
print('model loaded from checkpoint:', best_acc_path)
v_loss, v_acc, conf = validate(valid_loader, model, valid_criterion, -1)

#best loss
print('Training complete. Now testing best loss')
model.load_state_dict(torch.load(best_loss_path))
print('model loaded from checkpoint:', best_loss_path)
test_loss, test_acc, test_conf = validate(valid_loader, model, valid_criterion, -1)