import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
import sys
import  os
from utils import *
from tqdm.autonotebook import tqdm
import torchvision as tv

class Trainer():
    def __init__(self,opt,model,optimizer,lr_schedule,train_data_loader,valid_data_loader=None, start_epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.start_epoch = start_epoch
        self.opt = opt

        self.cur_epoch = start_epoch

        self.netd, self.netg, self.nets = model
        self.optimizer_d, self.optimizer_g, self.optimizer_s = optimizer
        self.scheduler_d, self.scheduler_g, self.scheduler_s = lr_schedule
        self.criterion=nn.BCELoss()
        self.contrast_criterion=nn.MSELoss()

    def train(self):
        if not os.path.exists(self.opt.work_dir):
            os.makedirs(self.opt.work_dir)
        true_labels=torch.ones(self.opt.batch_size)
        fake_labels=torch.zeros(self.opt.batch_size)
        if self.opt.use_gpu:
            self.criterion.cuda()
            self.contrast_criterion.cuda()
            true_labels,fake_labels=true_labels.cuda(),fake_labels.cuda()
            for epoch in range(self.opt.max_epoch):
                progressbar=tqdm(self.train_data_loader)
                d_loss=AverageMeter()
                g_loss=AverageMeter()
                c_loss=AverageMeter()
                s_loss=AverageMeter()
                for ii,(imgs,_) in enumerate(progressbar):
                    normal,defect,target=imgs
                    if self.opt.use_gpu:
                        normal=normal.cuda()
                        defect=defect.cuda()
                        target=target.cuda()
                    if (ii+1)%self.opt.d_every==0:
                        self.netd.train()
                        self.optimizer_d.zero_grad()
                        output=self.netd(normal)
                        error_d_real=self.criterion(output,true_labels)
                        error_d_real.backward()

                        fake_imgs=self.netg(defect).detach()
