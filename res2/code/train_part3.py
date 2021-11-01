from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time
import numpy as np
from numpy import *
from pathlib2 import Path
from data_loader.dataset import train_dataset, colorize_mask, fast_hist
# from models.u_net import UNet
# from models.seg_net import Segnet
# from tensorboardX import SummaryWriter
from csf_res2net import CSFNet
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))
cudnn.benchmark = True


def train():
    is_flip = 1
    size_w = 128
    size_h = 128
    inputchannle = 3
    save_epoch = 5
    tot_epoch = 60
    batch_size = 4
    data_path = "./part3_new/"
    model_save = "/tmp/data/part3"
    train_datatset_ = train_dataset(data_path, size_w, size_h, is_flip, inputchannle)
    train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=batch_size, shuffle=True)

    net = CSFNet(2, inputchannle, "../temp_data/pretrainmodel/res2net50_v1b_26w_4s-3cf99910.pth")
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    ###########   GLOBAL VARIABLES   ###########
    # initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
    # semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
    # initial_image = Variable(initial_image)
    # semantic_image = Variable(semantic_image)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255, weight=torch.FloatTensor([1, 6]).cuda())
    Path(model_save).mkdir(parents=True, exist_ok=True)
    start = time.time()
    net.train()
    for epoch in range(1, tot_epoch + 1):
        now_cal_map = 0
        for i, (initial_image_, semantic_image_, name) in enumerate(train_loader):
            initial_image_ = initial_image_.cuda()
            semantic_image_ = semantic_image_.cuda()

            semantic_image_pred = net(initial_image_)

            loss = criterion(semantic_image_pred, semantic_image_.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            now_cal_map += initial_image_.size()[0]
            # tot_union = np.sum(np.logical_or(predictions > 0.5, semantic_image.detach().cpu().numpy() > 0.5))
            # tot_inter = np.sum(np.logical_and(predictions > 0.5, semantic_image.detach().cpu().numpy() > 0.5))
            ########### Logging ##########
            print('[%d/%d][%d/%d] Loss: %.4f' %
                  (epoch, tot_epoch, now_cal_map, len(train_datatset_), loss.item()))

        if epoch % save_epoch == 0:
            torch.save(net.state_dict(), str(Path(model_save).joinpath('netG_' + str(epoch) + '.pth')))

    end = time.time()
    torch.save(net.state_dict(), str(Path(model_save).joinpath('netG_final.pth')))
    print('Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')


if '__main__' == __name__:
    train()
