# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse
import os

import torch.utils.model_zoo as model_zoo

C = edict()
config = C
cfg = C

C.seed = 12345

"""please config ROOT_dir and user when u first using"""
'''
C.repo_name = 'DLC'
C.abs_dir = osp.realpath(".")
C.net_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath(osp.join('/home/taiyan/DLC/log',
                                 C.net_dir))
C.result_dir = osp.join(C.log_dir, 'vals')
C.log_dir_link = osp.join(C.abs_dir, 'log')
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.result_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.result_dir + '/val_last.log'
'''
"""Path Config"""

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


#add_path(osp.join(C.root_dir, 'furnace'))

"""Image Config"""
C.num_classes = 19
C.background = -1
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
# 864 584
C.image_height = 512
C.image_width = 512
C.num_train_imgs = 20000
C.num_eval_imgs = 500
C.num_sample = 32
""" Settings for network, this would be different for each kind of model"""
C.fix_bias = True
C.fix_bn = False
C.sync_bn = True
# 新层的初始化
C.bn_eps = 1e-5
C.bn_momentum = 0.1
# C.pretrained_model = "/home/xin/tx/projects/TorchSeg/model_zoo/pytorch_model/resnet50_v1c_stem64.pth"
C.pretrained_model = "/home/xin/tx/projects/TorchSeg/model_zoo/pytorch_model/resnet18_v1.pth"
# C.pretrained_model = "/home/ying/data2/tx/TorchSeg/log/pspnet/cs.baseline.R18_v1c/snapshot/epoch-239.pth"

"""Train Config"""
C.lr = 5e-3
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4
C.batch_size = 12#*2 # 4 * C.num_gpu 32
C.nepochs = 16
C.niters_per_epoch = 1250
C.num_workers = 8
C.aux_loss_ratio = 0.4

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_batch_size = 1
C.eval_crop_size = 512
C.f = 2.1875 * 256

"""Display Config"""
C.snapshot_iter = 10
C.record_info_iter = 20
C.display_iter = 50


def open_tensorboard():
    pass

if __name__ == '__main__':
    print(config.epoch_num)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tb', '--tensorboard', default=False, action='store_true')
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
