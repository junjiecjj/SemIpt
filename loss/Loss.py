
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""
import sys,os
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()
from importlib import import_module

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#内存分析工具
from memory_profiler import profile
import objgraph

from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True



fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)


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
        self.CompressRateTrain = args.CompressRateTrain
        self.SNRtrain = args.SNRtrain

        self.losslog = torch.Tensor()

        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel( self.loss_module, range(args.n_GPUs) )

        # TODO
        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

        print(color.fuchsia(f"\n#============================ LOSS 准备完毕 ==============================\n"))

    #@profile
    def forward(self, sr, hr):
        # print(f"我正在计算loss\n")
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.losslog[-1, i] += effective_loss.item()  # tensor.item()  获取tensor的数值
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

    def mean_log(self, n_batches):
        self.losslog[-1].div_(n_batches)
        return self.losslog[-1]

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.losslog[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c/n_samples))

        return ''.join(log)

    # 在同一个画布中画出所有Loss的结果
    def plot_AllLoss(self, apath):
        fig, axs = plt.subplots(len(self.loss),1, figsize=(12,8))

        if len(self.loss) == 1:
            epoch = len(self.losslog)
            X = np.linspace(1, epoch, epoch)
            label = '{} Loss'.format(self.loss[0]['type'])

            axs.plot(X, self.losslog[:, 0].numpy(), label=label)
            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            axs.set_xlabel('Epochs', fontproperties=font)
            axs.set_ylabel(label, fontproperties=font)
            axs.set_title(label, fontproperties=font)
            #axs.grid(True)

            #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
            #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove SemiLight Nerd Font Complete Mono.otf", size=20)
            #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Light Nerd Font Complete.otf", size=20)
            legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
            frame1 = legend1.get_frame()
            frame1.set_alpha(1)
            frame1.set_facecolor('none')  # 设置图例legend背景透明

            axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
            axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
            axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
            axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

            axs.tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
            labels = axs.get_xticklabels() + axs.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
            [label.set_fontsize(20) for label in labels]  # 刻度值字号
        else:
            for i, l in enumerate(self.loss):
                epoch = len(self.losslog[:,i])
                X = np.linspace(1, epoch, epoch)
                label = '{} Loss'.format(l['type'])

                axs[i].plot(X, self.losslog[:, i].numpy(), label=label)
                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i].set_xlabel('Epochs', fontproperties=font)
                axs[i].set_ylabel(label, fontproperties=font)
                axs[i].set_title(label, fontproperties=font)
                axs[i].grid(True)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
                legend1 = axs[i].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                axs.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                axs.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                axs.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

                axs[1].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
                labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels]  # 刻度值字号

        fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离
        plt.tight_layout()#  使得图像的四周边缘空白最小化
        out_fig = plt.gcf()
        out_fig.savefig(os.path.join(apath, 'AllTrainLossPlot.pdf'))
        plt.show()
        plt.close(fig)
        return


    # 在不同画布画出不同loss的图，每张图的不同压缩率和信噪比单独画
    def plotAllLossSeperatedInOneFig(self, apath):
        part = len(self.CompressRateTrain)*len(self.SNRtrain)
        L = len(self.losslog)
        epoch = L // part
        X = np.linspace(1, epoch, epoch)
        for k, l in enumerate(self.loss):
            Label = '{} Loss'.format(self.loss[k]['type'])
            fig, axs = plt.subplots(len(self.SNRtrain), len(self.CompressRateTrain), figsize=(4*len(self.CompressRateTrain), 3*len(self.SNRtrain) ))
            print(f"axs.shape = {axs.shape}")
            for i, compr in enumerate(self.CompressRateTrain):
                for j, snr in enumerate(self.SNRtrain):
                    label = 'CompRatio={},SNR={}'.format(compr, snr)
                    start = (i*len(self.SNRtrain)+j)*epoch
                    end = (i*len(self.SNRtrain)+j+1)*epoch
                    Y = self.losslog[start:end, k].numpy()
                    axs[j,i].plot(X, Y, label=label)
                    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 12)
                    axs[j,i].set_xlabel('Epochs', fontproperties=font)
                    axs[j,i].set_ylabel(f"{l['type']}", fontproperties=font)
                    axs[j,i].set_title(label, fontproperties=font)

                    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                    font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Nerd Font Complete.otf", size=10)
                    font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=10)
                    legend1 = axs[j,i].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                    frame1 = legend1.get_frame()
                    frame1.set_alpha(1)
                    frame1.set_facecolor('none')  # 设置图例legend背景透明
    
                    axs[j,i].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                    axs[j,i].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                    axs[j,i].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                    axs[j,i].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
    
                    axs[j,i].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
                    labels = axs[j,i].get_xticklabels() + axs[j,i].get_yticklabels()
                    [label.set_fontname('Times New Roman') for label in labels]
                    [label.set_fontsize(10) for label in labels]  # 刻度值字号
            fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离
            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 13)
            plt.suptitle(f"{Label}", x=0.5, y=1, fontproperties=font4,)
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(os.path.join(apath, f"AllTrain{self.loss[k]['type']}LossPlot.pdf"))
            plt.show()
            plt.close(fig)
        return

    # 在不同画布画出不同loss的图，每张图的不同压缩率和信噪比单独画
    def plotAllLossSeperatedInOneFig1(self, apath):
        part = len(self.CompressRateTrain)*len(self.SNRtrain)
        L = len(self.losslog)
        epoch = L // part
        X = np.linspace(1, epoch, epoch)
        for k, l in enumerate(self.loss):
            Label = '{} Loss'.format(self.loss[k]['type'])
            fig, axs = plt.subplots( len(self.CompressRateTrain), len(self.SNRtrain), figsize=(4*len(self.SNRtrain), 3*len(self.CompressRateTrain)))
            print(f"axs.shape = {axs.shape}")
            for i, compr in enumerate(self.CompressRateTrain):
                for j, snr in enumerate(self.SNRtrain):
                    label = 'CompRatio={},SNR={}'.format(compr, snr)
                    start = (i*len(self.SNRtrain)+j)*epoch
                    end = (i*len(self.SNRtrain)+j+1)*epoch
                    Y = self.losslog[start:end, k].numpy()
                    axs[i,j].plot(X, Y, label=label)
                    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 12)
                    axs[i,j].set_xlabel('Epochs', fontproperties=font)
                    axs[i,j].set_ylabel(f"{l['type']}", fontproperties=font)
                    axs[i,j].set_title(label, fontproperties=font)

                    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                    font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Nerd Font Complete.otf", size=10)
                    font1 = FontProperties(fname=fontpath2+"Caskaydia Cove Regular Nerd Font Complete Mono.otf", size=10)
                    legend1 = axs[i,j].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                    frame1 = legend1.get_frame()
                    frame1.set_alpha(1)
                    frame1.set_facecolor('none')  # 设置图例legend背景透明
    
                    axs[i,j].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                    axs[i,j].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                    axs[i,j].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                    axs[i,j].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细
    
                    axs[i,j].tick_params(direction='in', axis='both', top=True,right=True,labelsize=16, width=3,)
                    labels = axs[i,j].get_xticklabels() + axs[i,j].get_yticklabels()
                    [label.set_fontname('Times New Roman') for label in labels]
                    [label.set_fontsize(10) for label in labels]  # 刻度值字号
            fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离
            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 13)
            plt.suptitle(f"{Label}", x=0.5, y=1, fontproperties=font4,)
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(os.path.join(apath, f"AllTrain{self.loss[k]['type']}LossPlot1.pdf"))
            plt.show()
            plt.close(fig)
        return


    # 在不同的画布中画各个损失函数的结果.
    def plot_loss(self, apath):
        epoch = len(self.losslog[:, 0])
        X = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(X, self.losslog[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'TrainLossPlot_{}.pdf'.format(l['type'])))
            plt.close(fig)
        return

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    # 在每个压缩率和信噪比下，所有的epoch训练完再调用保存
    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'TrainLossState.pt'))
        torch.save(self.losslog, os.path.join(apath, 'TrainLossLog.pt'))
        return


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
        return

# # # test LOSS module
# from torch.autograd import Variable
# los = LOSS(args,ckp)

# CompressRate = [0.17,0.33]
# SNR = [0,2,4,6,8,10]

# for cp_idx, CP in enumerate(CompressRate):
#     for snr_idx, snr in enumerate(SNR):
#         for epoch_idx in range(100):
#             los.start_log()
#             for batch in range(20):
#                 sr = torch.randn(1,3,4,4)
#                 hr = torch.randn(1,3,4,4)
#                 lss = los(sr, hr)
#                 lss = Variable(lss, requires_grad = True)
#                 lss.backward()
#             #los.end_log(10)
# los.plotAllLossSeperatedInOneFig1("/home/jack/snap/")
# los.plotAllLossSeperatedInOneFig("/home/jack/snap/")

# # los.plot_loss(ckp.dir,)
# los.plot_AllLoss(ckp.dir,)


# los.save(ckp.dir)
