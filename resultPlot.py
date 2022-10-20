
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
matplotlib.use('TkAgg')
#matplotlib.use('Agg')
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
from option import args
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
    def __init__(self, args, now):
        self.args = args
        if now == '2022-10-14-09:56:05':  # 2022-10-12-17:38:12  '2022-10-14-09:56:05'
            self.args.data_test=['Set2', 'Set3']
            self.args.CompressRateTrain = [0.17, 0.33]
            self.args.SNRtrain = [8, 10]
            self.args.SNRtest = [8, 10, 12, 14, 16, 18]

        if now == '2022-10-12-17:38:12':
            self.args.data_test=['Set5', 'Set14', 'B100', 'Urban100']
            self.args.CompressRateTrain = [0.17]
            self.args.SNRtrain = [8]
            self.args.SNRtest = [-2,0,2,4,6,8,10,12,14,16,18]

        self.marksize = 8
        self.home = '/home/jack'
        #self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        #self.now = '2022-10-14-09:56:05'
        self.savedir = self.home + '/snap/test/'

        os.makedirs(self.savedir, exist_ok=True)

        # self.load = self.home + f'/IPT-Pretrain-del/results/{now}'

        # self.trainloaddir = os.path.join(self.load + f"_TrainLog_{args.modelUse}")
        # self.testloaddir = os.path.join(self.trainloaddir, f"{now}")

        # if os.path.isfile(self.getTrainResPath('TrainMetric_log.pt')):
        #     print(f"存在训练结果文件：{self.getTrainResPath('TrainMetric_log.pt')}")
        #     self.metricLog = torch.load(self.trainloaddir+'/TrainMetric_log.pt')

        # if os.path.isfile(self.getTrainResPath('TrainLossLog.pt')):
        #     print(f"存在训练Loss文件：{self.getTrainResPath('TrainLossLog.pt')}")
        #     self.losslog = torch.load(self.trainloaddir+'/TrainLossLog.pt')

        # if os.path.isfile(self.getTestResPath('TestMetric_log.pt')):
        #     print(f"存在测试结果文件：{self.getTrainResPath('TrainMetric_log.pt')}")
        #     self.TeMetricLog = torch.load(self.getTestResPath('TestMetric_log.pt'))
        self.args.CompressRateTrain = [0.17,0.33,0.5]
        self.args.SNRtrain = [4, 8, 12, 18]
        self.args.SNRtest = [-2,0,2,4,6,8,10,12,14,16,18]


        dir1 = "/home/jack/IPT-Pretrain-del/results/"
        now1 = "2022-10-17-19:31:30"  # SNRTrain = 4dB,   R = 0.17
        now2 = "2022-10-12-17:38:12"  # SNRTrain = 8dB,   R = 0.17
        #now3 =  ""                    # SNRTrain = 12dB, R = 0.17
        now4 = "2022-10-14-15:44:52"  # SNRTrain = 18dB,  R = 0.17

        now5 = "2022-10-16-10:34:02"  # SNRTrain = 8dB,  R = 0.33
        now6 = "2022-10-16-23:59:01"  # SNRTrain = 8dB,  R = 0.5


        self.TeMetricLog1 = torch.load(dir1+f"{now1}_TrainLog_IPT/{now1}/TestMetric_log.pt")
        self.TeMetricLog2 = torch.load(dir1+f"{now2}_TrainLog_IPT/{now2}/TestMetric_log.pt")
        #self.TeMetricLog3 = torch.load(dir1+f"{now3}_TrainLog_IPT/{now3}/TestMetric_log.pt")
        self.TeMetricLog4 = torch.load(dir1+f"{now4}_TrainLog_IPT/{now4}/TestMetric_log.pt")
        self.TeMetricLog5 = torch.load(dir1+f"{now5}_TrainLog_IPT/{now5}/TestMetric_log.pt")
        self.TeMetricLog6 = torch.load(dir1+f"{now6}_TrainLog_IPT/{now6}/TestMetric_log.pt")
        self.TeMetricLog = {**self.TeMetricLog1, **self.TeMetricLog2, **self.TeMetricLog4,**self.TeMetricLog5,**self.TeMetricLog6}

        self.metricLog1 = torch.load(dir1+f"{now1}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog2 = torch.load(dir1+f"{now2}_TrainLog_IPT/TrainMetric_log.pt")
        #self.metricLog3 = torch.load(dir1+f"{now3}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog4 = torch.load(dir1+f"{now4}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog5 = torch.load(dir1+f"{now5}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog6 = torch.load(dir1+f"{now6}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog = {**self.metricLog1, **self.metricLog2, **self.metricLog4, **self.metricLog5, **self.metricLog6}


    def getTestResPath(self, *subdir):
        return os.path.join(self.testloaddir, *subdir)

    def getTrainResPath(self, *subdir):
        return os.path.join(self.trainloaddir, *subdir)

    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)
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
                fig = plt.figure()# constrained_layout=True
                tmpS = "MetricLog:CompRatio={},SNR={}".format(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
                # label = 'R={},SNR={}(dB)'.format(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
                label = r'$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$'%(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
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
                font1 = {'family':'Times New Roman','style':'normal','size':15}
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
                fig, axs=plt.subplots(len(self.args.SNRtrain),len(self.args.CompressRateTrain), figsize=(figWidth,figHigh), )# constrained_layout=True

                if len(self.args.SNRtrain) == 1 or len(self.args.CompressRateTrain) == 1:
                    axs = axs.reshape(len(self.args.SNRtrain), len(self.args.CompressRateTrain))

                for comprate_idx, comprateTmp in enumerate(self.args.CompressRateTrain):
                    for snr_idx, snrTmp in enumerate(self.args.SNRtrain):
                        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
                        #label = 'R={},SNR={}(dB)'.format(comprateTmp, snrTmp)
                        label = r'$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$'%(comprateTmp, snrTmp)

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
                        font1 = {'family':'Times New Roman','style':'normal','size':14}
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
            plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.4, hspace=0.2)
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(self.getSavePath(f"Train_{met}_EachRandSNREpoch_Plot.pdf"))
            out_fig.savefig(self.getSavePath(f"Train_{met}_EachRandSNREpoch_Plot.pdf"))
            plt.show()
            plt.close(fig)
        return



    """
    两张大图；每张图对应一个指标，PSNR或者MSE，下面以PSNR为例；
    每张图有很多条曲线；
    每条曲线画出在指定压缩率和信噪比下训练时PSNR随着epoch变化曲线；
    """
    #@profile
    def plot_AllTrainMetric1(self):
        mark  = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#000000','#0000FF', '#FF1493', '#ADFF2F', '#00FFFF','#FF8C00','#00FF00',  '#FF0000','#1E90FF','#808000', '#C000C0', '#800080']
        linestyles = ['-','--','-.',':']
        width = 6
        high = 4
        figWidth =width*len(self.args.CompressRateTrain)
        figHigh = high*len(self.args.SNRtrain)

        for idx, met in  enumerate(self.args.metrics):
            fig = plt.figure()# constrained_layout=True
            for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                for snrIdx, snr in enumerate(self.args.SNRtrain):
                    tmpS = "MetricLog:CompRatio={},SNR={}".format(compratio, snr)
                    label = r'$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$'%(compratio, snr)
                    epoch = len(self.metricLog[tmpS])
                    X = np.linspace(1, epoch, epoch)
                    idx1 = crIdx*len(self.args.SNRtrain)+snrIdx
                    plt.plot(X, self.metricLog[tmpS][:,idx], label=label,color=color[idx1],linestyle=linestyles[idx1],)

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            plt.xlabel('Epoch',fontproperties=font)
            if met=="PSNR":
                plt.ylabel(f"{met}(dB)",fontproperties=font)
            else:
                plt.ylabel(f"{met}",fontproperties=font)
            #axs[snr_idx,comprate_idx].set_title(label, fontproperties=font)

            #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=18)
            font1 = {'family':'Times New Roman','style':'normal','size':18}
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


        #============================================================================================================
            #plt.subplots_adjust(top=0.90, hspace=0.2)#调节两个子图间的距离
            plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.4, hspace=0.2)
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(self.getSavePath(f"Train_{met}_R_SNR_Epoch_PlotInOneFig.pdf"))
            out_fig.savefig(self.getSavePath(f"Train_{met}_R_SNR_Epoch_PlotInOneFig.eps"))
            plt.show()
            plt.close(fig)
        return



# <<< 训练结果画图
    """
    功能是：
    每张图对应一个数据集在不同压缩率下的PSNR/MSE-epoch曲线；
    每个数据集有两张图，每个图对应一个指标；
    对每张图有多条曲线，每一条曲线对应一个压缩率下的PSNR或MSE随SNR的变化曲线；
    """
    def PlotTestMetricSeperate(self):
        mark  = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#808000','#C000C0', '#000000','#00FFFF','#0000FF', '#FF1493', '#ADFF2F','#FF8C00','#00FF00', '#800080', '#FF0000','#1E90FF']

        high = 5
        width = 6

        for dsIdx, dtset in enumerate(self.args.data_test):
            for metIdx, met in enumerate(self.args.metrics):
                # fig, axs = plt.subplots(raw, col, figsize=(col*Len, raw*Len),constrained_layout=True)
                fig = plt.figure(figsize=(width, high),)# constrained_layout=True
                IDX = 0
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    for snrIdx, snr in enumerate(self.args.SNRtrain):
                        #tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dtset, compratio)
                        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio, snr)
                        data = self.TeMetricLog[tmpS]
                        # lb = f"R={compratio}"
                        lb = r"$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, snr)
                        plt.plot(data[:,0], data[:,metIdx+1], linestyle='-',color=color[IDX],marker=mark[IDX], label = lb)
                        IDX += 1
                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                plt.xlabel(r'$\mathrm{SNR}_\mathrm{test}\mathrm{(dB)}$',fontproperties=font)
                if met=="PSNR":
                    plt.ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    plt.ylabel(f"{met}",fontproperties=font)
                # plt.title(f"{dtset} dataset",loc = 'center',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
                font1 = {'family':'Times New Roman','style':'normal','size':16}
                legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                ax=plt.gca()#获得坐标轴的句柄
                ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


                plt.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号


                font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                plt.suptitle(f"{dtset} dataset",  fontproperties=font4,)

                # plt.subplots_adjust(top=0.86, wspace=0.2, hspace=0.2)#调节两个子图间的距离
                plt.tight_layout(pad=0.5, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
                #plt.subplots_adjust(top=0.89,bottom=0.01, left=0.01, right=0.99, wspace=0.4, hspace=0.2)

                out_fig = plt.gcf()
                out_fig.savefig(self.getSavePath(f"Test_{dtset}_{met}_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
                out_fig.savefig(self.getSavePath(f"Test_{dtset}_{met}_Plot.eps"), bbox_inches = 'tight',pad_inches = 0.2)
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
        alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        high = 5
        width = 6
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
            fig, axs = plt.subplots(raw, col, figsize=(col*width, raw*high),)# constrained_layout=True
            if raw == 1 and col==2:
                axs = axs.reshape(raw,col)
            for dsIdx, dtset in enumerate(self.args.data_test):
                i = dsIdx // col
                j = dsIdx % col
                IDX = 0
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    for snrIdx, snr in enumerate(self.args.SNRtrain):
                        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio,snr)
                        data = self.TeMetricLog[tmpS]
                        lb = r"$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, snr)
                        axs[i,j].plot(data[:,0], data[:,metIdx+1], linestyle='-', color=color[IDX], marker=mark[IDX], label = lb)
                        IDX += 1

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i,j].set_xlabel(r'$\mathrm{SNR}_\mathrm{test}\mathrm{(dB)}$',fontproperties=font)
                if met=="PSNR":
                    axs[i,j].set_ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    axs[i,j].set_ylabel(f"{met}",fontproperties=font)
                axs[i,j].set_title(f"{alabo[dsIdx]} {dtset}",loc = 'left',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
                font1 = {'family':'Times New Roman','style':'normal','size':14}
                legend1 = axs[i,j].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                axs[i,j].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                axs[i,j].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                axs[i,j].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                axs[i,j].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


                axs[i,j].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = axs[i,j].get_xticklabels() + axs[i,j].get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号

            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            plt.suptitle(f"{met} on " + ', '.join(self.args.data_test),  fontproperties=font4,)

            plt.tight_layout(pad=1.5 , h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
            #plt.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99, wspace=0.2, hspace=0.2)#调节两个子图间的距离
            out_fig = plt.gcf()
            out_fig.savefig(self.getSavePath(f"Test_{met}_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            out_fig.savefig(self.getSavePath(f"Test_{met}_Plot.eps"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return





    """
    两张大图；每张图对应一个指标，PSNR或者MSE，下面以PSNR为例；
    每张图有很多条曲线；
    每条曲线画出在指定压缩率和信噪比下训练时PSNR随着epoch变化曲线；
    """
    #@profile
    def plot_AllTrainMetricArodic(self):
        mark  = ['s','v', '^', '<', '>', '1', '2', '3', '4', '8',  'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#000000','#0000FF', '#FF1493', '#ADFF2F', '#00FFFF','#FF8C00','#00FF00',  '#FF0000','#1E90FF','#808000', '#C000C0', '#800080']
        linestyles = ['-','--','-.',':', '-','--','-.',':','-','--','-.',':']
        width = 6
        high = 4
        figWidth =width*len(self.args.CompressRateTrain)
        figHigh = high*len(self.args.SNRtrain)

        for idx, met in  enumerate(self.args.metrics):
            fig = plt.figure()# constrained_layout=True
            IDX = 0
            for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                for snrIdx, snr in enumerate(self.args.SNRtrain):
                    tmpS = "MetricLog:CompRatio={},SNR={}".format(compratio, snr)
                    if tmpS in self.metricLog.keys():
                        label = r'$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$'%(compratio, snr)
                        epoch = len(self.metricLog[tmpS])
                        X = np.linspace(1, epoch, epoch)
                        plt.plot(X, self.metricLog[tmpS][:,idx], label=label,color=color[IDX],linestyle=linestyles[IDX])
                        IDX += 1

            font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            plt.xlabel('Epoch',fontproperties=font)
            if met=="PSNR":
                plt.ylabel(f"{met}(dB)",fontproperties=font)
            else:
                plt.ylabel(f"{met}",fontproperties=font)
            #axs[snr_idx,comprate_idx].set_title(label, fontproperties=font)

            #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=18)
            font1 = {'family':'Times New Roman','style':'normal','size':14}
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


        #============================================================================================================
            #plt.subplots_adjust(top=0.90, hspace=0.2)#调节两个子图间的距离
            plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.4, hspace=0.2)
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            file = f"Train_{met}_R_SNR_Epoch_PlotInOneFig_R=({'-'.join([str(R) for R in self.args.CompressRateTrain])})_SNRtrain=({'-'.join([str(R) for R in self.args.SNRtrain])})"
            out_fig.savefig(self.getSavePath(file+".pdf"))
            out_fig.savefig(self.getSavePath(file+".eps"))
            plt.show()
            plt.close(fig)
        return



# <<< 训练结果画图
    """
    功能是：
    每张图对应一个数据集在不同压缩率下的PSNR/MSE-epoch曲线；
    每个数据集有两张图，每个图对应一个指标；
    对每张图有多条曲线，每一条曲线对应一个压缩率下的PSNR或MSE随SNR的变化曲线；
    """
    def PlotTestMetricSeperateArodic(self):
        mark  = ['s','v', '^', '<', '>', '1', '2', '3', '4', '8',  'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#808000','#C000C0', '#000000','#00FFFF','#0000FF', '#FF1493', '#ADFF2F','#FF8C00','#00FF00', '#800080', '#FF0000','#1E90FF']

        high = 5
        width = 6

        for dsIdx, dtset in enumerate(self.args.data_test):
            for metIdx, met in enumerate(self.args.metrics):
                # fig, axs = plt.subplots(raw, col, figsize=(col*Len, raw*Len),constrained_layout=True)
                fig = plt.figure(figsize=(width, high),)# constrained_layout=True
                IDX = 0
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    for snrIdx, snr in enumerate(self.args.SNRtrain):
                        #tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dtset, compratio)
                        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio, snr)
                        if tmpS in self.TeMetricLog.keys():
                            data = self.TeMetricLog[tmpS]
                            # lb = f"R={compratio}"
                            lb = r"$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, snr)
                            plt.plot(data[:,0], data[:,metIdx+1], linestyle='-',color=color[IDX],marker=mark[IDX],  markersize=self.marksize, label = lb)
                            IDX += 1
                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                plt.xlabel(r'$\mathrm{SNR}_\mathrm{test}\mathrm{(dB)}$',fontproperties=font)
                if met=="PSNR":
                    plt.ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    plt.ylabel(f"{met}",fontproperties=font)
                # plt.title(f"{dtset} dataset",loc = 'center',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
                font1 = {'family':'Times New Roman','style':'normal','size':14}
                legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                ax=plt.gca()#获得坐标轴的句柄
                ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


                plt.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号


                font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                plt.suptitle(f"{dtset} dataset",  fontproperties=font4,)

                # plt.subplots_adjust(top=0.86, wspace=0.2, hspace=0.2)#调节两个子图间的距离
                plt.tight_layout(pad=0.5, h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
                #plt.subplots_adjust(top=0.89,bottom=0.01, left=0.01, right=0.99, wspace=0.4, hspace=0.2)

                out_fig = plt.gcf()
                file = f"Test_{dtset}_{met}_Plot_R=({'-'.join([str(R) for R in self.args.CompressRateTrain])})_SNRtrain=({'-'.join([str(R) for R in self.args.SNRtrain])})"
                out_fig.savefig(self.getSavePath(file+".pdf"), bbox_inches = 'tight',pad_inches = 0.2)
                out_fig.savefig(self.getSavePath(file+".eps"), bbox_inches = 'tight',pad_inches = 0.2)
                plt.show()
                plt.close(fig)
        return

    """
    功能是：每张大图对应一个指标，即PSNR或MSE;
    每张大图画出所有的数据集的在不同压缩率下的PSNR/MSE随SNR的变化曲线；
    每张大图的各个子图对应各个数据集；
    每个子图对应一个数据集，每个子图有很多曲线，每一条曲线对应一个压缩率下的PSNR随SNR的变化曲线；
    """
    def PlotTestMetricInOneFigArodic(self):
        mark  = ['s','v', '^', '<', '>', '1', '2', '3', '4', '8',  'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#808000','#C000C0', 'red','cyan','blue','green','#FF8C00','#00FF00', '#FFA500', '#FF0000','#1E90FF']
        alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        high = 5
        width = 6
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
            fig, axs = plt.subplots(raw, col, figsize=(col*width, raw*high),)# constrained_layout=True
            if raw == 1 and col==2:
                axs = axs.reshape(raw,col)
            for dsIdx, dtset in enumerate(self.args.data_test):
                i = dsIdx // col
                j = dsIdx % col
                IDX = 0
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    for snrIdx, snr in enumerate(self.args.SNRtrain):
                        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio,snr)
                        if tmpS in self.TeMetricLog.keys():
                            data = self.TeMetricLog[tmpS]
                            lb = r"$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, snr)
                            axs[i,j].plot(data[:,0], data[:,metIdx+1], linestyle='-', color=color[IDX], marker=mark[IDX], markersize=self.marksize, label = lb)
                            IDX += 1

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i,j].set_xlabel(r'$\mathrm{SNR}_\mathrm{test}\mathrm{(dB)}$',fontproperties=font)
                if met=="PSNR":
                    axs[i,j].set_ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    axs[i,j].set_ylabel(f"{met}",fontproperties=font)
                axs[i,j].set_title(f"{alabo[dsIdx]} {dtset}",loc = 'left',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
                font1 = {'family':'Times New Roman','style':'normal','size':14}
                legend1 = axs[i,j].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                axs[i,j].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                axs[i,j].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                axs[i,j].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                axs[i,j].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


                axs[i,j].tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = axs[i,j].get_xticklabels() + axs[i,j].get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号

            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
            plt.suptitle(f"{met} on " + ', '.join(self.args.data_test),  fontproperties=font4,)

            plt.tight_layout(pad=1.5 , h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
            #plt.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99, wspace=0.2, hspace=0.2)#调节两个子图间的距离
            out_fig = plt.gcf()
            file = f"Test_{met}_Plot_R=({'-'.join([str(R) for R in self.args.CompressRateTrain])})_SNRtrain=({'-'.join([str(R) for R in self.args.SNRtrain])})"
            out_fig.savefig(self.getSavePath(file+".pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            out_fig.savefig(self.getSavePath(file+".eps"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return



# choice =  2

# if choice == 1:
#     now = '2022-10-14-09:56:05'
# elif choice ==2:
now = '2022-10-12-17:38:12'


pl = ResultPlot(args, now) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'

# pl.PlotTestMetricInOneFig()
# pl.PlotTestMetricSeperate()
# ##pl.plot_AllTrainMetric()
# pl.plot_AllTrainMetric1()




pl.PlotTestMetricInOneFigArodic()
pl.PlotTestMetricSeperateArodic()
# ##pl.plot_AllTrainMetric()
pl.plot_AllTrainMetricArodic()