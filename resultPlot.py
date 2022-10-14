
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

import os
import sys
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue
from torch.autograd import Variable
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import numpy as np
import imageio
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import collections
from torch.utils.tensorboard import SummaryWriter
from transformers import optimization


#内存分析工具
from memory_profiler import profile
import objgraph
import gc


# 本项目自己编写的库

sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

fontpath = "/usr/share/fonts/truetype/windows/"
# fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font = FontProperties(fname=fontpath+"simsun.ttf", size=22)


fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size=22)


fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"
font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)



# 功能：
class ResultPlot():
    def __init__(self, args):
        self.args = args
        self.home = '/home/jack'
        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.savedir = self.home + '/snap/test/'

        os.makedirs(self.savedir, exist_ok=True)

        self.load = self.home + f'/IPT-Pretrain/results/{self.now}'

        self.trainloaddir = os.path.join(self.load + f"_TrainLog_{args.modelUse}")
        self.testloaddir = os.path.join(self.trainloaddir, f"{self.now}")

        self.metricLog = torch.load(self.trainloaddir + 'TrainMetric_log.pt')
        self.TeMetricLog = torch.load(self.testloaddir + 'TestMetric_log.pt')


# >>> 训练结果画图

    """
    两张大图；每张图对应一个指标，PSNR或者MSE，下面以PSNR为例；
    每张图有len(self.args.SNRtrain)xlen(self.args.CompressRateTrain)个子图；
    每个子图对应在指定压缩率和信噪比下训练时PSNR随着epoch变化曲线；
    """
    #@profile
    def plot_AllTrainMetric(self):

        width = 6
        high = 4
        figWidth =width*len(self.args.CompressRateTrain)
        figHigh = high*len(self.args.SNRtrain)

        for idx, met in  enumerate(self.args.metrics):

            # 如果信噪比和压缩率只有一个
            if len(self.args.SNRtrain) == 1 and len(self.args.CompressRateTrain) == 1:
                fig = plt.figure(constrained_layout=True)
                tmpS = "MetricLog:CompRatio={},SNR={}".format(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
                label = 'R={},SNR={}(dB)'.format(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
                epoch = len(self.metricLog[tmpS])
                X = np.linspace(1, epoch, epoch)
                plt.plot(X, self.metricLog[tmpS][:,idx],'r-',label=label,)

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                plt.xlabel('Epoch',fontproperties=font)
                if met=="PSNR":
                    plt.ylabel(f"{met}(dB)",fontproperties=font)
                else:
                    plt.ylabel(f"{met}",fontproperties=font)
                #axs[snr_idx,comprate_idx].set_title(label, fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=18)
                legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                ax=plt.gca()#获得坐标轴的句柄
                ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

                plt.tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号
            else:
                fig, axs=plt.subplots(len(self.args.SNRtrain),len(self.args.CompressRateTrain), figsize=(figWidth,figHigh), constrained_layout=True)

                if len(self.args.SNRtrain) == 1 or len(self.args.CompressRateTrain) == 1:
                    axs = axs.reshape(len(self.args.SNRtrain), len(self.args.CompressRateTrain))

                for comprate_idx, comprateTmp in enumerate(self.args.CompressRateTrain):
                    for snr_idx, snrTmp in enumerate(self.args.SNRtrain):
                        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
                        label = 'R={},SNR={}(dB)'.format(comprateTmp, snrTmp)

                        epoch = len(self.metricLog[tmpS])
                        X = np.linspace(1, epoch, epoch)

                        axs[snr_idx,comprate_idx].plot(X, self.metricLog[tmpS][:,idx],'r-',label=label,)

                        font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                        axs[snr_idx,comprate_idx].set_xlabel('Epoch',fontproperties=font)
                        if met=="PSNR":
                            axs[snr_idx,comprate_idx].set_ylabel(f"{met}(dB)",fontproperties=font)
                        else:
                            axs[snr_idx,comprate_idx].set_ylabel(f"{met}",fontproperties=font)
                        #axs[snr_idx,comprate_idx].set_title(label, fontproperties=font)

                        #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                        font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=18)
                        legend1 = axs[snr_idx,comprate_idx].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                        frame1 = legend1.get_frame()
                        frame1.set_alpha(1)
                        frame1.set_facecolor('none')  # 设置图例legend背景透明

                        axs[snr_idx,comprate_idx].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                        axs[snr_idx,comprate_idx].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                        axs[snr_idx,comprate_idx].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                        axs[snr_idx,comprate_idx].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

                        axs[snr_idx,comprate_idx].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
                        labels = axs[snr_idx,comprate_idx].get_xticklabels() + axs[snr_idx,comprate_idx].get_yticklabels()
                        [label.set_fontname('Times New Roman') for label in labels]
                        [label.set_fontsize(20) for label in labels] #刻度值字号

            #plt.subplots_adjust(top=0.90, hspace=0.2)#调节两个子图间的距离
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(self.getSavePath(f"Train_{met}_EachRandSNREpoch_Plot.pdf"))
            plt.show()
            plt.close(fig)
        return


    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)


# <<< 训练结果画图
    """
    功能是：
    每张图对应一个数据集在不同压缩率下的PSNR/MSE-epoch曲线；
    每个数据集有两张图，每个图对应一个指标；
    对每张图有多条曲线，每一条曲线对应一个压缩率下的PSNR或MSE随SNR的变化曲线；
    """
    def PlotTestMetricSeperate(self):
        mark  = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#808000','#C000C0', '#000000','#00FFFF','#0000FF', '##FF1493', '#ADFF2F','#FF8C00','#00FF00', '##800080', '#FF0000','#1E90FF']

        Len = 6
        raw = 1
        col = 2

        for dsIdx, dtset in enumerate(self.args.data_test):
            for metIdx, met in enumerate(self.args.metrics):
                fig, axs = plt.subplots(raw, col, figsize=(col*Len, raw*Len),constrained_layout=True)
                fig = plt.figure(constrained_layout=True)
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dtset, compratio)
                    data = self.TeMetricLog[tmpS]
                    lb = f"R={compratio}"
                    axs.plot(data[:,0], data[:,metIdx+1], linestyle='-',color=color[crIdx],marker=mark[crIdx], label = lb)

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs.set_xlabel('SNR (dB)',fontproperties=font)
                if met=="PSNR":
                    axs.set_ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    axs.set_ylabel(f"{met}",fontproperties=font)
                axs.set_title(f"{dtset} dataset",loc = 'center',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
                legend1 = axs.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                ax=plt.gca()#获得坐标轴的句柄
                ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                axs.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


                plt.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号


                #font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                #plt.suptitle(f"{met} on " + ', '.join(self.args.data_test), x=0.5, y=0.98, fontproperties=font4,)

                # plt.subplots_adjust(top=0.86, wspace=0.2, hspace=0.2)#调节两个子图间的距离
                plt.tight_layout(pad=1, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化

                out_fig = plt.gcf()
                out_fig.savefig(self.getSavePath(f"Test_{met}_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
                plt.show()
                plt.close(fig)
        return

    """
    功能是：每张大图对应一个指标，即PSNR或MSE;
    每张大图画出所有的数据集的在不同压缩率下的PSNR/MSE随SNR的变化曲线；
    每张大图的各个子图对应各个数据集；
    每个子图对应一个数据集，每个子图有很多曲线，每一条曲线对应一个压缩率下的PSNR随SNR的变化曲线；
    """
    def PlotTestMetricInOneFig(self):
        mark  = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#808000','#C000C0', 'red','cyan','blue','green','#FF8C00','#00FF00', '#FFA500', '#FF0000','#1E90FF']

        Len = 6
        if len(self.args.data_test) > 2:
            raw = 2
            if len(self.args.data_test)%2 == 0:
                col = int(len(self.args.data_test)//2)
            else:
                col = int(len(self.args.data_test)//2)+1
        elif len(self.args.data_test) == 1:
            raw = 1
            col = 1
        elif len(self.args.data_test) == 2:
            raw = 1
            col = 2

        for metIdx, met in enumerate(self.args.metrics):
            fig, axs = plt.subplots(raw, col, figsize=(col*Len, raw*Len),constrained_layout=True)
            if raw == 1 and col==2:
                axs = axs.reshape(raw,col)
            for dsIdx, dtset in enumerate(self.args.data_test):
                i = dsIdx // col
                j = dsIdx % col
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dtset, compratio)
                    data = self.TeMetricLog[tmpS]
                    lb = f"R={compratio}"
                    axs[i,j].plot(data[:,0], data[:,metIdx+1], linestyle='-',color=color[crIdx],marker=mark[crIdx], label = lb)

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i,j].set_xlabel('SNR (dB)',fontproperties=font)
                if met=="PSNR":
                    axs[i,j].set_ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    axs[i,j].set_ylabel(f"{met}",fontproperties=font)
                axs[i,j].set_title(f"{dtset}",loc = 'right',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
                legend1 = axs[i,j].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                axs[i,j].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                axs[i,j].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                axs[i,j].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                axs[i,j].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

                fontt2 = {'family':'Times New Roman','style':'normal','size':16}
                legend1 = axs[i,j].legend(loc='best',borderaxespad=0,edgecolor='black',prop=fontt2,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none') # 设置图例legend背景透明

                axs[i,j].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = axs[i,j].get_xticklabels() + axs[i,j].get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号

            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            plt.suptitle(f"{met} on " + ', '.join(self.args.data_test), x=0.5, y=0.98, fontproperties=font4,)

            # plt.subplots_adjust(top=0.86, wspace=0.2, hspace=0.2)#调节两个子图间的距离
            plt.tight_layout(pad=1, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化

            out_fig = plt.gcf()
            out_fig.savefig(self.getSavePath(f"Test_{met}_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return



# from option import args

# ckp = checkpoint(args)
# # 依次遍历压缩率
# for comprate_idx, compressrate in enumerate(args.CompressRateTrain):  #[0.17, 0.33, 0.4]
#     # 依次遍历信噪比
#     for snr_idx, snr in enumerate(args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
#         #print(f"\nNow， Train on comprate_idx = {comprate_idx}, compressrate = {compressrate}， snr_idx = {snr_idx}, snr = {snr}, \n")

#         epoch = 0

#         ckp.InitMetricLog(compressrate, snr)
#         # 遍历epoch
#         for epoch_idx in  range(100):
#             ckp.UpdateEpoch()
#             epoch += 1
#             #初始化特定信噪比和压缩率下的存储字典
#             ckp.AddMetricLog(compressrate, snr)

#             # 遍历训练数据集
#             for i in range(20):
#                 # pass
#                 ckp.UpdateMetricLog(compressrate, snr, epoch_idx+i)
#             ckp.MeanMetricLog(compressrate, snr, 20)

# #ckp.plot_trainPsnr(0.4, 18)
# ckp.save()


# <<< 测试相关函数




# model = net()
# LR = 0.01
# opt = make_optimizer(args,model,100)
# loss = torch.nn.CrossEntropyLoss()

# lr_list1 = []
# lr_list2 = []
# for epoch in range(200):
#       for i in range(20):
#           y = torch.randint(0, 9, (10,10))*1.0
#           opt.zero_grad()
#           out = model(torch.randn(10,1))
#           lss = loss(out, y)
#           lss = Variable(lss, requires_grad = True)
#           lss.backward()
#           opt.step()
#       opt.schedule()
#       lr_list2.append(opt.get_lr())
#       lr_list1.append(opt.state_dict()['param_groups'][0]['lr'])

# fig, axs = plt.subplots(1,1, figsize=(6,6))
# axs.plot(range(len(lr_list1)),lr_list1,color = 'r')
# #plt.plot(range(100),lr_list2,color = 'b')
# out_fig = plt.gcf()
# out_fig.savefig("/home/jack/snap/11.pdf")
# plt.show()
# plt.close(fig)


# from option import args
# ckp = checkpoint(args)
# ckp.InittestDir('aaaa')
# for idx_data, ds in enumerate(args.data_test):
#     for comprate_idx, compressrate in enumerate(args.CompressRateTrain):
#         ckp.InitTestMetric(compressrate, ds)
#         for snr_idx, snr in enumerate( args.SNRtest):
#             ckp.AddTestMetric(compressrate, snr, ds)
#             for i in range(20):
#                 metric = torch.tensor([comprate_idx,comprate_idx+snr_idx])
#                 ckp.UpdateTestMetric(compressrate, ds,metric)
#                 ckp.MeanTestMetric(compressrate, ds,2)

# ckp.PlotTestMetric()

#  使用时：
"""

model = net()
LR = 0.01
optimizer = make_optimizer( args, model, )


lr_list1 = []

for epoch in range(200):
    for X,y in dataloder:
        optimizer.step()
    optimizer.schedule()
    lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(200),lr_list1,color = 'r')

plt.show()

"""
