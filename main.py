# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
from typing import Optional
from util.logger import setup_logger

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

#import datasets
import util.misc as utils
#from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_DABDETR, build_dab_deformable_detr
from util.utils import clean_state_dict

from datasets import WireframeDatasetA
from datasets import ScanNetDatasetA
from dataloader import get_train_loader_su3
from dataloader import TrainPre
import cv2
from tqdm import tqdm
from glob import glob
from pylsd.lsd import lsd


def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)
    
    # about lr
    parser.add_argument('--lr', default=5e-5, type=float, 
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, 
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=220, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--save_checkpoint_interval', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--modelname', '-m', type=str, choices=['dab_detr', 'dab_deformable_detr'], default='dab_deformable_detr')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int, 
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, 
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, 
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true', 
                        help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int, 
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int, 
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', 
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true', 
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int, 
                        help="number of deformable attention sampling points in encoder layers")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, 
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float, 
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float, 
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float, 
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, 
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float, 
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25, 
                        help="alpha for focal loss")


    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true', 
                        help="Using for debug only. It will fix the size of input images to the maximum.")


    # Traing utils
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+', 
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true', 
                        help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true', 
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true', 
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser


def build_model_main(args):
    if args.modelname.lower() == 'dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    elif args.modelname.lower() == 'dab_deformable_detr':
        model, criterion, postprocessors = build_dab_deformable_detr(args)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)
    # torch.autograd.set_detect_anomaly(True)
    
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="DAB-DETR")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config.json")
        # print("args:", vars(args))
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    '''
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    '''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))


    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    data_loader_train, sampler_train = get_train_loader_su3(WireframeDatasetA)
    '''
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    '''
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 5e-7)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,200], gamma=0.32)

    '''
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    '''
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            #lr_scheduler.gamma = 0.1
            #optimizer.param_groups[0]['lr'] = 5e-6
            #optimizer.param_groups[1]['lr'] = 1e-6
    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
        # import ipdb; ipdb.set_trace()

    '''
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return
    '''
    print("Start training")
    start_time = time.time()
    best_aa = 0.0
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for epoch in range(args.start_epoch, args.epochs):
        #for epoch in range(1):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        
        print(epoch)
        if epoch>=0:
            the =  0.1
            model.eval()
            with torch.no_grad():
                results_list = []
                classAcc_list = []
                inlier_list = []
                for img, mat in tqdm(zip(img_list, mat_list)):
                    img = img[:, :, ::-1]
                    img = normalize(img, np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)

                    samples = img
                    targets = mat
                    samples = samples.to(device)
                    outputs = model(samples)
        
                    mat = np.expand_dims(mat, 0)
                    prob = outputs['pred_logits'][:,:,0].sigmoid()
                    
                    pred_vps = outputs['pred_boxes'][0].cpu()
                    idx = prob.argsort(descending=True)[0].cpu()
                    
                    vps = torch.zeros((1,3,3))
                    '''
                    vps = torch.zeros((1,400,3))
                    for i in range(400):
                        vps[0,i,:] = pred_vps[i] / torch.linalg.norm(pred_vps[i])
                    
                    '''
                    vps[0,0,:] = pred_vps[idx[0]] / torch.linalg.norm(pred_vps[idx[0]])
                    
                    for i in range(1, len(idx)):
                        cos1 = cos(vps[0,0,:], pred_vps[idx[i]] / torch.linalg.norm(pred_vps[idx[i]]))
                        if cos1 <= the and cos1 >=-the:
                            j = i
                            vps[0,1,:] = pred_vps[idx[j]] / torch.linalg.norm(pred_vps[idx[j]])
                            break
                            
                    for i in range(j+1, len(idx)):
                        cos2 = cos(vps[0,0,:], pred_vps[idx[i]]/ torch.linalg.norm(pred_vps[idx[i]]))
                        cos3 = cos(vps[0,1,:], pred_vps[idx[i]]/ torch.linalg.norm(pred_vps[idx[i]]))
                        if cos2 <= the and cos2 >= -the and cos3 <= the and cos3>=-the:
                            k = i
                            vps[0,2,:] = pred_vps[idx[k]] / torch.linalg.norm(pred_vps[idx[k]])
                            break
                    
                    
                    '''
                    with open(r"./now.txt", "a") as f:
                        f.write(str(idx[0])+' '+ str(prob[0,idx[0]]) + "\n")
                        f.write(str(idx[j])+' '+ str(prob[0,idx[j]]) + "\n")
                        f.write(str(idx[k])+' '+ str(prob[0,idx[k]]) + "\n")
                        #print(id, score[idx])
                    print(prob[0,idx[0]], prob[0,idx[j]], prob[0,idx[k]])
                    #print(vps,mat)
                    '''
                    
                    #print(vps)
                    #vps = select(ori, vps, lengths)
                    results = get_results(vps, mat)
                    #results = get_results2(vps, mat, prob)
                    results_list = results_list + results
        
                results_list = np.sort(np.array(results_list))
                ave = compute_score(results_list)
                y = (1 + np.arange(len(results_list))) / len(results_list)
                aa = [f"{AA(results_list, y, th):.3f}" for th in [0.5, 1, 2, 3, 5, 10, 20]]
                print(aa)
                print(ave)
                now_aa = float(aa[3])
                if now_aa > best_aa:
                    outputname = 'checkpoint'+str(int(now_aa * 1000))+'.pth'
                    best_aa = now_aa
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, output_dir / outputname)
                    #np.savez('err_su3_ours_C3_20.npz', err=results_list)
                if now_aa >= best_aa:
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(' '.join(aa)+ "\n")
                            
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_beforedrop.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        '''
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        '''
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     #**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        epoch_time = time.time() - epoch_start_time
        #epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        #log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            '''
            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
            '''
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))


def exchange(mat):
    # for su3 dataset
    for i in range(3):
        if mat[2,i] < 0:
            mat[:,i] = -mat[:,i]
    newmat = np.zeros((3,3))
    newmat[:,0] = mat[:,2]
    x = mat[0,:]
    i = np.argmin(x)
    newmat[:,1] = mat[:,i]
    i = np.argmax(x)
    newmat[:, 2] = mat[:, i]
    return newmat


def vps2vectors(vps_list):
    vectors_list = []
    for vps in vps_list:
        if vps is None:
            vectors_list.append(None)
            continue
        vectors = []
        for vp in vps:
            vectors.append(vp2vector(vp))
        vectors_list.append(vectors)
    return vectors_list


def vp2vector(vp):
    vp = np.concatenate((vp - [[256], [256]], np.array([[config.f]])),axis=0)
    vector = vp / np.linalg.norm(vp, 2)
    return vector


def get_results(preds, gts):
    results = []
    for pred, gt in zip(preds, gts):
        if pred is None:
            continue
        for i in range(3):
            results.append(min(np.arccos(np.abs(pred[i] @ gt).clip(max=1))) / np.pi * 180)
    return results

def get_results2(preds, gts):
    results = []
    
    for pred, gt in zip(preds, gts):
        if pred is None:
            continue
        for g in range(3):
            gt_now = gt[:,g:g+1]
            results.append(min(np.arccos(np.abs(pred @ gt_now).clip(max=1)))[0] / np.pi * 180)
        
    return results
    
def get_results3(preds, gts, scores):
    results = []
    
    for pred, gt,score in zip(preds, gts, scores):
        if pred is None:
            continue
        for g in range(3):
            gt_now = gt[:,g:g+1]
            angle = min(np.arccos(np.abs(pred @ gt_now).clip(max=1)))[0]
            idx = np.argmin(np.arccos(np.abs(pred @ gt_now).clip(max=1)))
            results.append(angle / np.pi * 180)
            with open(r"./idea.txt", "a") as f:
                f.write(str(idx)+' '+ str(score[idx]) + "\n")
            print(idx, score[idx])
        
    return results
    
def compute_score(results_list):
    s = 0
    c = 0
    for si in results_list:
        c += 1
        s += si
    return s / c

def AA(x, y, threshold):
    index = np.searchsorted(x, threshold)
    x = np.concatenate([x[:index], [threshold]])
    y = np.concatenate([y[:index], [threshold]])
    return ((x[1:] - x[:-1]) * y[:-1]).sum() / threshold


def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def select(oris, vps, lengths):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    the = 0.01
    vps_ori = vps[0]
    vps = vps[0].cpu().numpy()
    d = np.abs(vps @ oris.T)
    fil = d < the
    sum_len = np.zeros((d.shape[0]))
    for i in range(d.shape[0]):
        sum_len[i] = - np.sum(lengths[fil[i]])
    idx = np.argsort(sum_len)
    
    
    vps_r = torch.zeros((1,3,3))
    vps_r[0,0] = vps_ori[idx[0]]
    
    the = 0.02
    for i in range(1, len(idx)):
       cos1 = cos(vps_r[0,0], vps_ori[idx[i]])
       if -the <= cos1 <= the:
           j = i
           vps_r[0,1] = vps_ori[idx[j]]
           break
                            
    for i in range(j+1, len(idx)):
        cos2 = cos(vps_r[0,0,:], vps_ori[idx[i]])
        cos3 = cos(vps_r[0,1,:], vps_ori[idx[i]])
        if -the <= cos2 <= the and -the <= cos3 <= the:
            k = i
            vps_r[0,2] = vps_ori[idx[k]]
            break
    
    return vps_r
        
        

if __name__ == '__main__':
    rootdir = "/home/ying/data3/datasets/su3/"
    filelist = sorted(glob(f"{rootdir}/*/*.png"))[:500]
    img_list = []
    line_list = []
    mat_list = []
    ori_list = []
    l_list = []
    f = 2.1875 * 256
    for name in tqdm(filelist):
        img = cv2.imread(name)
        img_list.append(img)
        with np.load(name.replace("_haze.png", "_label.npz")) as npz:
            mat = npz['vpts']
            mat = mat.T
            mat[1, :] = -mat[1, :]
            mat = exchange(mat)
            mat_list.append(mat)
        '''
        with np.load(name.replace(".png", "_linesAll.npz")) as npz:
            lines = npz['lines']
            lines = lines[:, 0:4]
            line_list.append(lines)
        
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)
        #lines = lsd(gray)
        
        p1 = np.stack((lines[:, 0] - 256, lines[:, 1] - 256, np.ones_like(lines[:, 0]) * f), axis=1)
        p2 = np.stack((lines[:, 2] - 256, lines[:, 3] - 256, np.ones_like(lines[:, 0]) * f), axis=1)
        l = np.sqrt(np.square(lines[:, 3] - lines[:, 1]) + np.square(lines[:, 2] - lines[:, 0]))
        
        #l = l[idx]
        oris = np.cross(p1, p2)
        oris = oris / np.linalg.norm(oris, 2, axis=1, keepdims=True)
        ori_list.append(oris)
        l_list.append(l)
        '''
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
