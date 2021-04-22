import os
import argparse
parser = argparse.ArgumentParser(description='Training regurgitation detection model')
parser.add_argument('--cuda', default='', type=str, metavar='cuda', help='export GPU to use')
parser.add_argument('--cfg', default='', type=str, metavar='PATH', help='path to load training config')
args = parser.parse_args()
# this one is to export GPU to use.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import logging
import numpy as np
np.set_printoptions(precision=4, suppress=True)

from config import  *
from datetime import datetime
from poly_scheduler import PolyScheduler
from setting import *
from tqdm import tqdm
from collections import OrderedDict, abc
from sync_batchnorm import convert_model
from setting import *
from convertions import tensor_label_to_one_hot
from metrics import rlt_table, MetricEvaluator

class Trainer:
    def __init__(self,
                 log_path,
                 config,
                 **kwargs
                 ):
        # params
        self.log_path = log_path
        self.config = config
        self.base_lr = [config.base_lr]
        self.momentum = 0.0
        self.alpha = 0
        self.model_key = config.model_key 
        self.resume_path = config.resume_path
        self.best_model_path = config.best_model_path
        self.total_epoch = config.total_epoch
        self.start_epoch = config.start_epoch
        self.weight_decay = config.weight_decay
        self.use_cuda = config.use_cuda
        self.optim_key = config.optim_key
        self.best_model_loss = float('inf')
        self.best_model_epoch = None
        self.model = None
        self.dataloader = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.train_params = None
        self.scheduler = None
        self.pred_loss = None
        self.aux_loss = None
        self.optimizer = None
        self.summary = None
        self.writer = None
        self.kwargs = kwargs

    def prepare(self):
        model_class = model_dict[self.config.model_class]
        self.model = model_class(**self.config.model_params.config_dict)

        if self.resume_path is not None:
            self.model, check_point = self._load_model(self.resume_path, model_dict[self.config.model_class], self.config.model_params.config_dict, self.model)
            self.best_model_loss = float('inf')
            if self.config.restart:
                self.start_epoch = 0
                self.best_model_loss = float('inf')
            else:
                self.start_epoch = check_point['epoch'] + 1
                self.best_model_loss = check_point['loss']

        if self.use_cuda:
            self.model = convert_model(self.model)
            self.model = nn.DataParallel(self.model)
            self.model.to("cuda")

        loader_class = loader_dict[self.config.loader_class]
        self.dataloader = loader_class(**self.config.loader_params.config_dict)

        self.train_loader = self.dataloader.train_loader
        self.valid_loader = self.dataloader.val_loader
        self.test_loader = self.dataloader.test_loader

        self.train_params = {
            'params': self.model.parameters(),
            'lr': self.base_lr[0],
            'weight_dacay': self.weight_decay
            },

        if self.config.optim_key == 'sgd':
            self.optimizer = optim.SGD(
                self.train_params,
                nesterov=False,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.config.optim_key == 'adam':
            self.optimizer = optim.Adam(self.train_params)
        else:
            raise RuntimeError('optim error')

        self.scheduler = PolyScheduler(self.base_lr, self.total_epoch, len(self.train_loader), warmup_epochs=0)
        self.pred_loss = loss_dict[self.config.loss_class](**config.loss_params.config_dict)

        self.seg_metric_board = metric_dict[self.config.seg_metric_class]
        self.seg_evaluator = MetricEvaluator(self.seg_metric_board)
        self.cls_metric_board = metric_dict[self.config.cls_metric_class]
        self.cls_evaluator = MetricEvaluator(self.cls_metric_board)
        self.kf_metric_board = metric_dict[self.config.kf_metric_class]
        self.kf_evaluator = MetricEvaluator(self.kf_metric_board)

        self.best_rlt = None

    def train(self, epoch):
        train_loss_epoch = 0
        total_sample_num = 0
        self.model.train()
        print('\n'+'='*178)
        print('[epoch: %d]' % (epoch))
        tbar = tqdm(self.train_loader, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}' + '[{elapsed}<{remaining}, {rate_inv_fmt}]')
        for i, batch in enumerate(tbar):
            x, seg_true, cls_true, kf_true = batch['video'], batch['mask'], batch['label'], batch['kf']
            x = x.cuda()
            cls_true = cls_true.cuda()
            seg_true = seg_true[kf_true].cuda()
            kf_true = kf_true.cuda()

            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()

            seg_pred, cls_pred, kf_pred = self.model(x, kf_true)
            pred_loss = self.pred_loss(seg_pred, cls_pred, kf_pred, seg_true, cls_true, kf_true)
            pred_loss.backward()
            self.optimizer.step()

            total_sample_num += 1
            train_loss_epoch += pred_loss.item()

            tbar.set_description('loss: %.3f' % (train_loss_epoch / total_sample_num))

        self.dataloader.train_dataset.update_feed_list()
        print('train_loss: ', train_loss_epoch / total_sample_num)
        self.writer.add_scalar('train/loss', train_loss_epoch / total_sample_num, epoch)
        print('-'*178)

    def validate(self, epoch):
        self.seg_evaluator.reset()
        self.cls_evaluator.reset()
        self.kf_evaluator.reset()
        self.model.eval()
        val_loss_epoch = 0
        total_sample_num = 0

        print('Validation:')
        tbar = tqdm(self.valid_loader, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}' + '[{elapsed}<{remaining}, {rate_inv_fmt}]')
        for _, batch in enumerate(tbar):
            x, seg_true, cls_true, kf_true  = batch['video'], batch['mask'], batch['label'], batch['kf']
            x = x.cuda()
            cls_true = cls_true.cuda()
            seg_true = seg_true[kf_true].cuda()
            kf_true = kf_true.cuda()

            with torch.no_grad():
                seg_pred, cls_pred, kf_pred = self.model(x, kf_true)
                pred_loss = self.pred_loss(seg_pred, cls_pred, kf_pred, seg_true, cls_true, kf_true)
                total_sample_num += 1
                val_loss_epoch += pred_loss.item()

                self.seg_evaluator.update(seg_pred[kf_true], seg_true[kf_true])
                self.cls_evaluator.update(cls_pred, tensor_label_to_one_hot(cls_true, 2))
                self.kf_evaluator.update(kf_pred[:,None], kf_true.view(-1, 1).float())
        print('val_loss: ', val_loss_epoch / total_sample_num)

        seg_rlt = self.seg_evaluator.eval()
        cls_rlt = self.cls_evaluator.eval()
        kf_rlt = self.kf_evaluator.eval()
        tb = rlt_table(seg_rlt, field_names=['Metric', 'Value @ epoch %d'%epoch])
        tb = rlt_table(cls_rlt, field_names=['Metric', 'Value @ epoch %d'%epoch], tb=tb)
        tb = rlt_table(kf_rlt, field_names=['Metric', 'Value @ epoch %d'%epoch], tb=tb)
        if self.best_rlt is not None:
            tb.add_column(fieldname='Value @ epoch %d (best checkpoint)'%self.best_model_epoch, column=list(self.best_rlt.values()))
        with open(os.path.join(self.log_path, 'val_rlt.txt'), 'a') as f:
            f.write(str(tb)+'\n')
        print(tb)

        # save model
        if val_loss_epoch / total_sample_num < self.best_model_loss:
            self.best_rlt = seg_rlt
            self.best_rlt.update(cls_rlt)
            self.best_rlt.update(kf_rlt)
            trainer_logger.info('Current model is the best model. Validation loss decreased from {:4f} to {:4f} at epoch {:d}'.format(self.best_model_loss, val_loss_epoch/total_sample_num, epoch))
            self.best_model_path = os.path.join(self.log_path, 'best_'+self.model_key+'.pth')
            torch.save({
                'epoch': self.start_epoch + epoch, 
                'model_state_dict': self.model.state_dict(),
                'loss': val_loss_epoch / total_sample_num
                }, self.log_path + '/best_' + self.model_key + '.pth')
            self.best_model_loss = val_loss_epoch / total_sample_num
            self.best_model_epoch = self.start_epoch + epoch
        torch.save({
            'epoch': self.start_epoch + epoch, 
            'model_state_dict': self.model.state_dict(),
            'loss': val_loss_epoch / total_sample_num
            }, self.log_path + '/last_' + self.model_key + '.pth')


    def test(self):
        self.seg_evaluator.reset()
        self.cls_evaluator.reset()
        self.kf_evaluator.reset()

        model_class = model_dict[self.config.model_class]
        self.model = model_class(**self.config.model_params.config_dict)
        self.model, _ = self._load_model(self.best_model_path, model_dict[self.config.model_class], self.config.model_params.config_dict, self.model)

        if self.use_cuda:
            self.model.to("cuda")
        self.model.eval()
        print('\nTest: ')
        tbar = tqdm(self.test_loader, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}' + '[{elapsed}<{remaining}, {rate_inv_fmt}]')
        total_sample_num = 0
        test_loss_epoch = 0.0

        for i, batch in enumerate(tbar):
            x, seg_true, cls_true, kf_true = batch['video'], batch['mask'], batch['label'], batch['kf']
            x = x.cuda()
            cls_true = cls_true.cuda()
            seg_true = seg_true[kf_true].cuda()
            kf_true = kf_true.cuda()

            with torch.no_grad():
                seg_pred, cls_pred, kf_pred = self.model(x, kf_true)
                pred_loss = self.pred_loss(seg_pred, cls_pred, kf_pred, seg_true, cls_true, kf_true)
                total_sample_num += 1
                test_loss_epoch += pred_loss.item()

                self.seg_evaluator.update(seg_pred[kf_true], seg_true[kf_true])
                self.cls_evaluator.update(cls_pred, tensor_label_to_one_hot(cls_true, 2))
                self.kf_evaluator.update(kf_pred[:,None], kf_true.view(-1, 1).float())
        print('test_loss: ', test_loss_epoch / total_sample_num)

        seg_rlt = self.seg_evaluator.eval()
        cls_rlt = self.cls_evaluator.eval()
        kf_rlt = self.kf_evaluator.eval()
        epoch = -1
        tb = rlt_table(seg_rlt, field_names=['Metric', 'Value @ epoch %d'%epoch])
        tb = rlt_table(cls_rlt, field_names=['Metric', 'Value @ epoch %d'%epoch], tb=tb)
        tb = rlt_table(kf_rlt, field_names=['Metric', 'Value @ epoch %d'%epoch], tb=tb)

        with open(os.path.join(self.log_path, 'test_rlt.txt'), 'a') as f:
            f.write(str(tb)+'\n')
        print(tb)

    def _load_model(self, model_path, model_class, model_params=None, model=None):
        if model is None:
            model = model_class(**model_params)
        check_point = torch.load(model_path)
        state_dict = check_point['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return(model, check_point)

def fix_randomness(numpy_seed):
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(numpy_seed)
    torch.manual_seed(numpy_seed)
    torch.cuda.manual_seed(numpy_seed)
    torch.cuda.manual_seed_all(numpy_seed)
    random.seed(numpy_seed)
    np.random.seed(numpy_seed)
    
if __name__ == "__main__":
    # config
    yaml_file = args.cfg
    print('Training configuration: \n')
    config = get_config(yaml_file)

    log_name = config.model_key + '_lr_' + str(config.base_lr) \
            + '_bs_' + str(config.loader_params.batch_size) + '_total_epoch_' + str(config.total_epoch) + \
            '_wdecay_' + str(config.weight_decay) + '_optim_' + str(config.optim_key)
    today = datetime.now()
    log_folder_name = today.strftime('%Y%m%d') + '{:02}'.format(today.hour) + log_name
    log_path = os.path.join(config.log_dir, config.experiment_name, log_folder_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    config.export(os.path.join(log_path, 'setting.yaml'))
    fix_randomness(100)

    # logger
    trainer_logger = logging.getLogger('unet_rcnn')
    handler = logging.FileHandler(os.path.join(log_path, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    trainer_logger.addHandler(handler)

    # train
    trainer_logger.info('Start training')
    trainer = Trainer(log_path, config)
    trainer.prepare()
    if config.mode == 'train':
        for epoch in range(trainer.start_epoch, trainer.total_epoch):
            trainer.train(epoch)
            trainer.validate(epoch)
        trainer.test()
    elif config.mode == 'test':
        trainer.test()
    trainer.writer.close()
