
# -*- coding: utf-8 -*-
"""
Created on 2022/07/31

@author: Junjie Chen

"""
import sys, os
sys.path.append("/home/jack/公共的/Pretrained-IPT-cjj/")
sys.path.append("..")
from option import args
from torch.utils.tensorboard import SummaryWriter
import os, sys
import torch



class SummWriter(SummaryWriter):
    def __init__(self, args):
        sdir = args.SummaryWriteDir
        os.makedirs(sdir, exist_ok=True)
        kwargs_summwrit = {'comment':"", 'purge_step': None, 'max_queue': 10, 'flush_secs':120,}
        super(SummWriter, self).__init__(sdir, **kwargs_summwrit)

    def WrTLoss(self, trainloss, epoch):
        self.add_scalar('train/Loss/allLoss', trainloss, epoch)

    def WrTrainLoss(self,  compratio, snr, trainloss, epoch):
        self.add_scalar('train/Loss/CompreRatio={},SNR={}'.format(compratio, snr), trainloss, epoch)

    def WrTrainPsnr(self, compratio, snr, trainPsnr, epoch):
        self.add_scalar('train/PSNR/CompreRatio={},SNR={}'.format(compratio, snr), trainPsnr, epoch)

    def WrModel(self, model, images ):
        self.add_graph(model, images)

    def WrClose(self):
        self.close()

wr = SummWriter(args)



# x = range(100)
# for i in x:
#     wr.add_scalar('Loss/train', i * 2, i)
# wr.close()


x = range(300,400)

AllEpoch = 0
for comprate_idx, compressrate in enumerate(args.CompressRateTrain):  #[0.17, 0.33, 0.4]
    for snr_idx, snr in enumerate(args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
        for epch_idx in range(200):
            AllEpoch += 1
            for batch_idx  in range(10):
                pass
            wr.WrTLoss( torch.tensor(comprate_idx+snr_idx+epch_idx+0.121), AllEpoch )
            wr.WrTrainLoss(compressrate, snr, torch.tensor(1.22+epch_idx), epch_idx)
            wr.WrTrainPsnr(compressrate, snr, torch.tensor(2.11+epch_idx), epch_idx)


wr.close()


# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# writer = SummaryWriter('/home/jack/IPT-Pretrain/results/summary')

# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train/1', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test/1', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train/2', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test/2', np.random.random(), n_iter)
# r = 5
# for i in range(100):
#     writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
#                                     'xcosx':i*np.cos(i/r),
#                                     'tanx': np.tan(i/r)}, i)


# writer.close()


















































