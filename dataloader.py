import cv2
import torch
import numpy as np
from torch.utils import data
import sys 
from config import config


class TrainPre(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img):
        # img, gt = random_mirror(img, gt)

        img = normalize(img, self.img_mean, self.img_std)

        img = img.transpose(2, 0, 1)

        extra_dict = None

        return img, extra_dict

def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def get_train_loader_scannet(engine, dataset):
    dataset_dir = "/home/ying/data2/scannet-vp/"

    train_preprocess = TrainPre(config.image_mean, config.image_std)

    train_dataset = dataset(dataset_dir, "train", train_preprocess, outlier=True)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // 4
        is_shuffle = False
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
    
def get_train_loader_su3(dataset, flip=False):
    dataset_dir = "/home/ying/data3/datasets/su3/"

    train_preprocess = TrainPre(config.image_mean, config.image_std)

    train_dataset = dataset(dataset_dir, "train", train_preprocess, flip)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler)

    return train_loader, train_sampler
