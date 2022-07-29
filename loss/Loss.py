
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""
import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LOSS(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(LOSS, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):  #  ['1*MSE']
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()

            self.loss.append({'type': loss_type, 'weight': float(weight), 'function': loss_function} )

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.losslog = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel( self.loss_module, range(args.n_GPUs) )

        # TODO
        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.losslog[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.losslog[-1, -1] += loss_sum.item()
        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        #  losslog.shape = [1,len(loss)],[2,len(loss)],[2,len(loss)]...,[epoch,len(loss)]
        self.losslog = torch.cat((self.losslog, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.losslog[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.losslog[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c/n_samples))

        return ''.join(log)

    def plot_loss(self, apath):
        epoch = self.losslog[:, 0]
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.losslog[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'TrainLossPlot_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    # 所有的epoch训练完再调用保存
    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'TrainLossState.pt'))
        torch.save(self.losslog, os.path.join(apath, 'TrainLossLog.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        if  os.path.isfile(os.path.join(apath,'TrainLossState.pt')):
            self.load_state_dict(torch.load(os.path.join(apath, 'TrainLossState.pt'), **kwargs))

        if  os.path.isfile(os.path.join(apath,'TrainLossLog.pt')):
            self.losslog = torch.load(os.path.join(apath, 'TrainLossLog.pt'))

        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.losslog)): l.scheduler.step()


# from torch.autograd import Variable
# los = LOSS(args,ckp)

# CompressRate = [1,2,3]
# SNR = [-10,-6,-2,0,2,6,10]

# for cp_idx, CP in enumerate(CompressRate):
#     for snr_idx, snr in enumerate(SNR):
#         for epoch_idx in range(20):
#             sr = torch.randn(1,3,4,4)
#             hr = torch.randn(1,3,4,4)
#             los.start_log()
#             lss = los(sr, hr)
#             lss = Variable(lss, requires_grad = True)
#             lss.backward()
#             los.end_log(10)



# los.plot_loss(ckp.dir, 420)
# los.save(ckp.dir)























