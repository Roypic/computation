# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:44
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : train.py
"""

"""
# from utils.networks import NestedUNet #import your network
import argparse
import torch
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='NestedUNet')
    parser.add_argument('--intputchannel',type=int,default=1)
    parser.add_argument('--outputchannel', type=int, default=1)
    configs = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network=eval(configs.mode)(configs.intputchannel,configs.outputchannel)
    network.to(device=device)
    print(count_param(network))
