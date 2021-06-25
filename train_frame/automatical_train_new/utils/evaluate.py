

import argparse
from PIL import Image
from torchvision import transforms
from pre_add_gt import confusion_image
import cv2
import logging
import os
import os.path as osp
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime, localtime
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.dataset import BasicDataset
from config import UNetConfig

from losses import LovaszLossSoftmax
from losses import LovaszLossHinge
from losses import dice_coeff
from judge import dice,iou




def eval_net(net, loader, device, n_val, cfg):
    """
    Evaluation without the densecrf with the dice coefficient

    """
    net.eval()
    tot = 0
    dice_score=0
    iou_score=0
    #inference_mask = torch.sigmoid(inference_masks) > cfg.out_threshold
    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # compute loss
            if cfg.deepsupervision:
                masks_preds = net(imgs)
                masks_preds=F.softmax(masks_preds, dim=1)
                loss = 0
                for masks_pred in masks_preds:
                    tot_cross_entropy = 0
                    for true_mask, pred in zip(true_masks, masks_pred):
                        pred = (pred > cfg.out_threshold).float()
                        if cfg.n_classes > 1:
                            sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        else:
                            sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        tot_cross_entropy += sub_cross_entropy
                    tot_cross_entropy = tot_cross_entropy / len(masks_preds)
                    tot += tot_cross_entropy
            else:
                masks_pred = net(imgs)
                for true_mask, pred in zip(true_masks, masks_pred):

                    if cfg.n_classes > 1:
                        pred = (torch.softmax(pred, dim=0) > cfg.out_threshold).float()
                        tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        dice_score+=dice(pred,true_mask)
                        iou_score+=iou(pred,true_mask)

                    else:
                        pred = (torch.sigmoid(pred) > cfg.out_threshold).float()

                        #(pred.shape,true_mask.shape)
                        tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        dice_score += dice(pred, true_mask)
                        iou_score += iou(pred, true_mask)


            pbar.update(imgs.shape[0])
    print("iou:",iou_score/n_val,"dice:",dice_score/n_val)
    return tot / n_val,iou_score/n_val,dice_score/n_val

def test_net(net, loader, device, n_val, cfg, result_path):
    """
    Evaluation without the densecrf with the dice coefficient

    """
    ftest = open(os.path.join(result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
    wrtest = csv.writer(ftest)
    net.eval()
    tot = 0
    dice_score=0
    iou_score=0
    counter = 0
    #inference_mask = torch.sigmoid(inference_masks) > cfg.out_threshold
    with tqdm(total=n_val, desc='test round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if cfg.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # compute loss
            if cfg.deepsupervision:
                masks_preds = net(imgs)
                masks_preds=F.softmax(masks_preds, dim=1)
                loss = 0
                for masks_pred in masks_preds:
                    tot_cross_entropy = 0
                    for true_mask, pred in zip(true_masks, masks_pred):
                        pred = (pred > cfg.out_threshold).float()
                        if cfg.n_classes > 1:
                            sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        else:
                            sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        tot_cross_entropy += sub_cross_entropy
                    tot_cross_entropy = tot_cross_entropy / len(masks_preds)
                    tot += tot_cross_entropy
            else:
                masks_pred = net(imgs)
                for true_mask, pred in zip(true_masks, masks_pred):

                    if cfg.n_classes > 1:
                        pred = (torch.softmax(pred, dim=0) > cfg.out_threshold).float()
                        tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1)).item()
                        dice_score+=dice(pred,true_mask)
                        iou_score+=iou(pred,true_mask)
                    else:
                        pred = (torch.sigmoid(pred) > cfg.out_threshold).float()
                        #(pred.shape,true_mask.shape)
                        tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                        dice_score += dice(pred, true_mask)
                        iou_score += iou(pred, true_mask)


            pbar.update(imgs.shape[0])
            if cfg.n_classes == 1:
                # writer.add_images('masks/true', batch_masks, global_step)
                inference_mask = torch.sigmoid(masks_pred) > cfg.out_threshold
                # print(inference_mask)

                output, outputmask = confusion_image(imgs, inference_mask, true_masks)
                for epochnum in range(len(output)):
                    outputpil = Image.fromarray(output[epochnum])
                    outputmsk = Image.fromarray(outputmask[epochnum].squeeze())
                    # print("test:", output[epochnum].shape)
                    outputpil.save(result_path + '/' + str(counter) + '.png')
                    outputmsk.save(result_path + '/' + str(counter) + 'mask.png')
                    counter = counter + 1
    print("testiou:", iou_score / n_val, "testdice:", dice_score / n_val)
    return tot / n_val,iou_score/n_val,dice_score/n_val