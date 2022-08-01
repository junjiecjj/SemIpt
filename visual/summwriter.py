
# -*- coding: utf-8 -*-
"""
Created on 2022/07/31

@author: Junjie Chen

"""

from option import args
from torch.utils.tensorboard import SummaryWriter
import os, sys
sys.path.append("/home/jack/公共的/Pretrained-IPT-cjj/")
sys.path.append("..")



class SummWriter(SummaryWriter):
    def __init__(self, args):
        sdir = args.SummaryWriteDir
        os.makedirs(sdir, exist_ok=True)
        kwargs_summwrit = {'comment':"", 'purge_step': None, 'max_queue': 10, 'flush_secs':120,}
        super(SummWriter, self).__init__(sdir, **kwargs_summwrit)

    def WrTrainLoss(self, trainloss, lossmodule, epoch):
        self.add_scalars('Loss/train', trainloss,epoch)

    def WrTrainPsnr(self, trainPsnr, compratio, snr, epoch):
        self.add_scalar('Psnr/train/{}-{}'.format(compratio, snr), trainPsnr, epoch)

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
for i in x:
    wr.add_scalar('Loss/train', i * 1.5, i)
wr.close()


# from torch.utils.tensorboard import SummaryWriter
# import numpy as np

# writer = SummaryWriter('/home/jack/IPT-Pretrain/results/summary')

# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)




















































