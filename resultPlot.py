
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen


mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]

可用颜色：
cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'



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
        self.legendsize = 16
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
        ## 1
        # self.args.CompressRateTrain = [0.17, 0.33, 0.5]
        # self.args.SNRtrain = [8]
        # self.savedir = self.home + '/snap/result/EpsPdf_SNR=8dB_R=(017_033_050)/'


        ## 2
        # self.args.CompressRateTrain = [0.17,]
        # self.args.SNRtrain = [4, 8, 12, 18]
        # self.savedir = self.home + '/snap/result/EpsPdf_SNR=(4,8,12,18)dB_R=017/'


        # ## 3
        # self.args.CompressRateTrain = [ 0.5]
        # self.args.SNRtrain = [4, 8, 12, 18]
        # #  self.savedir = self.home + '/snap/result/EpsPdf_SNR=(4,8,12,18)dB_R=050/'
        # self.savedir = self.home + '/snap/test/'


        # ## 4
        # self.args.CompressRateTrain = [0.17, 0.33, 0.5]
        # self.args.SNRtrain = [18]
        # self.savedir = self.home + '/snap/result/EpsPdf_SNR=18dB_R=(017_033_050)/'
        # self.savedir = self.home + '/snap/test/'

        ## 5
        # self.args.CompressRateTrain = [0.17, 0.33, 0.5]
        # self.args.SNRtrain = [4, 8, 12, 18]
        # #self.savedir = self.home + '/snap/result/EpsPdf_SNR=(4,8,12,18)dB_R=(017-033-050)/'
        # self.savedir = self.home + '/snap/test/'

        self.args.SNRtest = [-2,0,2,4,6,8,10,12,14,16,18]


        dir1 = "/home/jack/IPT-Pretrain-del/results/"
        now1 = "2022-10-17-19:31:30"  # SNRTrain = 4dB,   R = 0.17
        now2 = "2022-10-12-17:38:12"  # SNRTrain = 8dB,   R = 0.17
        now3 = "2022-10-20-09:59:33" # SNRTrain = 12dB,  R = 0.17
        now4 = "2022-10-14-15:44:52"  # SNRTrain = 18dB,  R = 0.17

        now5 = "2022-10-16-10:34:02"  # SNRTrain = 8dB,   R = 0.33
        now6 = "2022-10-16-23:59:01"  # SNRTrain = 8dB,   R = 0.5

        now7 = "2022-10-21-10:38:50"  # SNRTrain = 18dB,  R = 0.5
        now8 = "2022-10-22-14:03:59"  # SNRTrain = 4 dB,  R = 0.5
        now9 = "2022-10-23-12:32:59"  # SNRTrain = 12 dB,  R = 0.5
        now0 = "2022-10-21-23:38:16"  # SNRTrain = 18 dB,  R = 0.33

        self.TeMetricLog1 = torch.load(dir1+f"{now1}_TrainLog_IPT/{now1}/TestMetric_log.pt")
        self.TeMetricLog2 = torch.load(dir1+f"{now2}_TrainLog_IPT/{now2}/TestMetric_log.pt")
        self.TeMetricLog3 = torch.load(dir1+f"{now3}_TrainLog_IPT/{now3}/TestMetric_log.pt")
        self.TeMetricLog4 = torch.load(dir1+f"{now4}_TrainLog_IPT/{now4}/TestMetric_log.pt")
        self.TeMetricLog5 = torch.load(dir1+f"{now5}_TrainLog_IPT/{now5}/TestMetric_log.pt")
        self.TeMetricLog6 = torch.load(dir1+f"{now6}_TrainLog_IPT/{now6}/TestMetric_log.pt")
        self.TeMetricLog7 = torch.load(dir1+f"{now7}_TrainLog_IPT/{now7}/TestMetric_log.pt")
        self.TeMetricLog8 = torch.load(dir1+f"{now8}_TrainLog_IPT/{now8}/TestMetric_log.pt")
        self.TeMetricLog9 = torch.load(dir1+f"{now9}_TrainLog_IPT/{now9}/TestMetric_log.pt")
        self.TeMetricLog0 = torch.load(dir1+f"{now0}_TrainLog_IPT/{now0}/TestMetric_log.pt")


        self.TeMetricLog = {**self.TeMetricLog1, **self.TeMetricLog2,**self.TeMetricLog3, **self.TeMetricLog4,**self.TeMetricLog5,**self.TeMetricLog6, **self.TeMetricLog7, **self.TeMetricLog8, **self.TeMetricLog9, **self.TeMetricLog0}

        self.metricLog1 = torch.load(dir1+f"{now1}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog2 = torch.load(dir1+f"{now2}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog3 = torch.load(dir1+f"{now3}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog4 = torch.load(dir1+f"{now4}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog5 = torch.load(dir1+f"{now5}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog6 = torch.load(dir1+f"{now6}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog7 = torch.load(dir1+f"{now7}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog8 = torch.load(dir1+f"{now8}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog9 = torch.load(dir1+f"{now9}_TrainLog_IPT/TrainMetric_log.pt")
        self.metricLog0 = torch.load(dir1+f"{now0}_TrainLog_IPT/TrainMetric_log.pt")

        self.metricLog = {**self.metricLog1, **self.metricLog2,**self.metricLog3, **self.metricLog4, **self.metricLog5, **self.metricLog6,**self.metricLog7, **self.metricLog8, **self.metricLog9, **self.metricLog0}


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
                        font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
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
            font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
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
                font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
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
                font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#CD5C5C','#00CED1','#7B68EE', '#0000FF', '#FF0000','#808000' ]
        linestyles = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',]
        width = 8
        high = 6
        self.legendsize = 16
        for idx, met in  enumerate(self.args.metrics):
            fig = plt.figure(figsize=(width, high),)# constrained_layout=True
            IDX = 0
            for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                for snrIdx, snr in enumerate(self.args.SNRtrain):
                    tmpS = "MetricLog:CompRatio={},SNR={}".format(compratio, snr)
                    if tmpS in self.metricLog.keys():
                        label = r'$\mathrm{R}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$'%(compratio, snr)
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
            font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
        high = 5
        width = 6
        self.legendsize = 14
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
                            lb = r"$\mathrm{R}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, snr)
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
                font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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

                if len(self.args.CompressRateTrain) == 1:
                    plt.suptitle(f"{dtset} dataset, "+'Fixed '+r"$\mathrm{R} = %.2f$"%(self.args.CompressRateTrain[0]) + ', vary the '+r'$\mathrm{SNR}_\mathrm{train}$', fontproperties=font4,)
                elif len(self.args.SNRtrain) == 1:
                    plt.suptitle(f"{dtset} dataset, "+'Fixed ' + r"$\mathrm{SNR}_\mathrm{train} = %d$"%self.args.SNRtrain[0] + "(dB), vary the "+r"$\mathrm{R}$", fontproperties=font4,)

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
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
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
                            lb = r"$\mathrm{R}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, snr)
                            axs[i,j].plot(data[:,0], data[:,metIdx+1], linestyle='-', color=color[IDX], marker=mark[IDX], markersize=self.marksize, label = lb)
                            IDX += 1

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i,j].set_xlabel(r'$\mathrm{SNR}_\mathrm{test}\mathrm{(dB)}$',fontproperties=font)
                if met=="PSNR":
                    axs[i,j].set_ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    axs[i,j].set_ylabel(f"{met}",fontproperties=font)
                axs[i,j].set_title(f"{alabo[dsIdx]} {dtset} dataset",loc = 'left',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
                font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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
            if len(self.args.CompressRateTrain) == 1:
                plt.suptitle('Fixed '+r"$\mathrm{R} = %.2f$"%(self.args.CompressRateTrain[0]) + ', vary the '+r'$\mathrm{SNR}_\mathrm{train}$', fontproperties=font4,)
            elif len(self.args.SNRtrain) == 1:
                plt.suptitle('Fixed ' + r"$\mathrm{SNR}_\mathrm{train} = %d$"%self.args.SNRtrain[0] + "(dB), vary the "+r"$\mathrm{R}$", fontproperties=font4,)

            plt.tight_layout(pad=1.5 , h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
            #plt.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99, wspace=0.2, hspace=0.2)#调节两个子图间的距离
            out_fig = plt.gcf()
            file = f"Test_{met}_Plot_R=({'-'.join([str(R) for R in self.args.CompressRateTrain])})_SNRtrain=({'-'.join([str(R) for R in self.args.SNRtrain])})"
            out_fig.savefig(self.getSavePath(file+".pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            out_fig.savefig(self.getSavePath(file+".eps"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return



    """
    功能是：每张大图对应一个指标，即PSNR或MSE;
    每张大图画出所有的数据集的在不同SNR下训练，然后测试时的PSNR/MSE随压缩率R的变化曲线；
    每张大图的各个子图对应各个数据集；
    每个子图对应一个数据集，每个子图有很多曲线，每一条曲线对应一个SNR训练和测试下的PSNR随R的变化曲线；
    """
    def PlotTestPSNR2RInOneFigArodic(self):
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
        alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

        self.savedir = self.home + '/snap/test/'

        self.R =  [0.0833, 0.17, 0.25, 0.33, 0.5]
        self.SNR = [0, 4, 8, 12]



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
                for snrIdx, snr in enumerate(self.SNR):
                    YPSNR = []
                    for crIdx, compratio in enumerate(self.R):
                        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio, snr)
                        if tmpS in self.keys:
                            data = self.TeMetricLog[tmpS]
                            YPSNR.append(data[:,metIdx+1])

                    lb = r"$\mathrm{SNR}=%d\mathrm{(dB)}$"%(snr)
                    axs[i,j].plot(self.R[:len(YPSNR)], YPSNR, linestyle='-', color=color[IDX], marker=mark[IDX], markersize=self.marksize, label = lb)
                    IDX += 1

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                axs[i,j].set_xlabel(r'$\mathrm{R}$',fontproperties=font)
                if met=="PSNR":
                    axs[i,j].set_ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    axs[i,j].set_ylabel(f"{met}",fontproperties=font)
                axs[i,j].set_title(f"{alabo[dsIdx]} {dtset} dataset",loc = 'left',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
                font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
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

            plt.suptitle('AWGN channel', fontproperties=font4,)

            plt.tight_layout(pad=1.5 , h_pad=1, w_pad=1.7)#  使得图像的四周边缘空白最小化
            #plt.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99, wspace=0.2, hspace=0.2)#调节两个子图间的距离
            out_fig = plt.gcf()
            file = f"Test_{met}_Plot_SNR=({'-'.join([str(R) for R in self.SNR])})_R=({'-'.join([str(R) for R in self.R])}))"
            out_fig.savefig(self.getSavePath(file+".pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            out_fig.savefig(self.getSavePath(file+".eps"), bbox_inches = 'tight',pad_inches = 0.2)
            out_fig.savefig(self.getSavePath(file+".png"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return




    def ReadResult(self):

        dir1 = "/home/jack/IPT-Pretrain-del/results/"
        self.savedir = self.home + '/snap/test/'

        now1 = "2022-10-24-15:52:05"  # SNRTrain = 0dB,   R = 0.083
        now2 = "2022-10-24-19:55:26"  # SNRTrain = 4dB,   R = 0.083
        now3 = "2022-10-25-08:50:27"  # SNRTrain = 8dB,   R = 0.083
        now4 = "2022-10-25-13:14:22"  # SNRTrain = 12dB,  R = 0.083

        now5 = "2022-10-26-16:09:40"  # SNRTrain = 0dB,   R = 0.17
        now6 = "2022-10-26-11:54:46"  # SNRTrain = 4dB,   R = 0.17
        now7 = "2022-10-26-09:04:28"  # SNRTrain = 8dB,   R = 0.17
        now8 = "2022-10-25-19:22:31"  # SNRTrain = 12dB,  R = 0.17

        now9 = "2022-10-26-19:15:26"   # SNRTrain = 0dB,   R = 0.25
        now10 = "2022-10-27-08:46:21"  # SNRTrain = 4dB,  R = 0.25
        now11 = "2022-10-30-10:37:53"   # 2022-10-27-12:01:50"  # SNRTrain = 8dB,  R = 0.25
        now12 = "2022-10-30-20:03:22" #"2022-10-30-14:15:59"# "2022-10-27-19:28:00"  # SNRTrain = 12dB, R = 0.25

        now13 = "2022-10-28-20:23:45"  # SNRTrain = 0dB,   R = 0.33
        now14 = "2022-10-28-14:39:15"  # SNRTrain = 4dB,   R = 0.33
        now15 = "2022-10-28-11:50:32"  # SNRTrain = 8dB,   R = 0.33
        now16 =  "2022-10-30-17:12:24" #"2022-10-28-08:48:31"  # SNRTrain = 12dB,  R = 0.33

        now17 = "2022-10-29-12:45:16"   # SNRTrain = 0dB,   R = 0.5
        now18 = "2022-10-29-15:41:41"   # SNRTrain = 4dB,   R = 0.5
        now19 = "2022-10-29-19:24:40"   # SNRTrain = 8dB,   R = 0.5
        now20 = "2022-10-29-22:05:56"  # SNRTrain = 12dB,  R = 0.5


        self.TeMetricLog1 = torch.load(dir1+f"{now1}_TrainLog_IPT/{now1}/TestMetric_log.pt")
        self.TeMetricLog2 = torch.load(dir1+f"{now2}_TrainLog_IPT/{now2}/TestMetric_log.pt")
        self.TeMetricLog3 = torch.load(dir1+f"{now3}_TrainLog_IPT/{now3}/TestMetric_log.pt")
        self.TeMetricLog4 = torch.load(dir1+f"{now4}_TrainLog_IPT/{now4}/TestMetric_log.pt")

        self.TeMetricLog5 = torch.load(dir1+f"{now5}_TrainLog_IPT/{now5}/TestMetric_log.pt")
        self.TeMetricLog6 = torch.load(dir1+f"{now6}_TrainLog_IPT/{now6}/TestMetric_log.pt")
        self.TeMetricLog7 = torch.load(dir1+f"{now7}_TrainLog_IPT/{now7}/TestMetric_log.pt")
        self.TeMetricLog8 = torch.load(dir1+f"{now8}_TrainLog_IPT/{now8}/TestMetric_log.pt")

        self.TeMetricLog9 = torch.load(dir1+f"{now9}_TrainLog_IPT/{now9}/TestMetric_log.pt")
        self.TeMetricLog10 = torch.load(dir1+f"{now10}_TrainLog_IPT/{now10}/TestMetric_log.pt")
        self.TeMetricLog11 = torch.load(dir1+f"{now11}_TrainLog_IPT/{now11}/TestMetric_log.pt")
        self.TeMetricLog12 = torch.load(dir1+f"{now12}_TrainLog_IPT/{now12}/TestMetric_log.pt")

        self.TeMetricLog13 = torch.load(dir1+f"{now13}_TrainLog_IPT/{now13}/TestMetric_log.pt")
        self.TeMetricLog14 = torch.load(dir1+f"{now14}_TrainLog_IPT/{now14}/TestMetric_log.pt")
        self.TeMetricLog15 = torch.load(dir1+f"{now15}_TrainLog_IPT/{now15}/TestMetric_log.pt")
        self.TeMetricLog16 = torch.load(dir1+f"{now16}_TrainLog_IPT/{now16}/TestMetric_log.pt")

        self.TeMetricLog17 = torch.load(dir1+f"{now17}_TrainLog_IPT/{now17}/TestMetric_log.pt")
        self.TeMetricLog18 = torch.load(dir1+f"{now18}_TrainLog_IPT/{now18}/TestMetric_log.pt")
        self.TeMetricLog19 = torch.load(dir1+f"{now19}_TrainLog_IPT/{now19}/TestMetric_log.pt")
        self.TeMetricLog20 = torch.load(dir1+f"{now20}_TrainLog_IPT/{now20}/TestMetric_log.pt")

        self.TeMetricLog = {**self.TeMetricLog1, **self.TeMetricLog2, **self.TeMetricLog3, **self.TeMetricLog4,**self.TeMetricLog5,**self.TeMetricLog6, **self.TeMetricLog7, **self.TeMetricLog8, **self.TeMetricLog9, **self.TeMetricLog10, **self.TeMetricLog11, **self.TeMetricLog12, **self.TeMetricLog13, **self.TeMetricLog14, **self.TeMetricLog15, **self.TeMetricLog16, **self.TeMetricLog17, **self.TeMetricLog18, **self.TeMetricLog19, **self.TeMetricLog20}

        self.keys  = self.TeMetricLog.keys()

        self.R =  [0.0833, 0.17, 0.25, 0.33, 0.5]
        self.SNR = [0, 4, 8, 12]

        return


    """
    功能是：每张大图对应一个指标，即PSNR或MSE;
    每张大图画出所有的数据集的在不同SNR下训练，然后测试时的PSNR/MSE随压缩率R的变化曲线；
    每张大图的各个子图对应各个数据集；
    每个子图对应一个数据集，每个子图有很多曲线，每一条曲线对应一个SNR训练和测试下的PSNR随R的变化曲线；
    """
    def PlotTestPSNR2RSeperateArodic(self):
        mark  = ['s','v','*', 'o', 'd', '>', '1', 'p', '2', 'h', 'P', '3', '|', 'X', '4', '8', 'H', '+', 'x', 'D',  '_']
        color = ['#1E90FF','#FF6347', '#800080', '#008000', '#FFA500', '#C71585', '#7FFF00', '#EE82EE' ,'#00CED1','#CD5C5C','#7B68EE', '#0000FF', '#FF0000','#808000' ]
        alabo = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

        self.R =  [0.0833, 0.17, 0.25, 0.33, 0.5]
        self.SNR = [0, 4, 8, 12]
        self.savedir = self.home + '/snap/test/'

        high = 5
        width = 6

        for metIdx, met in enumerate(self.args.metrics):
            for dsIdx, dtset in enumerate(self.args.data_test):
                fig = plt.figure(figsize=(width, high),)# constrained_layout=True
                IDX = 0
                for snrIdx, snr in enumerate(self.SNR):
                    YPSNR = []
                    for crIdx, compratio in enumerate(self.R):
                        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio, snr)
                        if tmpS in self.keys:
                            data = self.TeMetricLog[tmpS]
                            YPSNR.append(data[:,metIdx+1])

                    lb = r"$\mathrm{SNR}=%d\mathrm{(dB)}$"%(snr)
                    plt.plot(self.R[:len(YPSNR)], YPSNR, linestyle='-', color=color[IDX], marker=mark[IDX], markersize=self.marksize, label = lb)
                    IDX += 1

                font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                plt.xlabel(r'$\mathrm{R}$',fontproperties=font)
                if met=="PSNR":
                    plt.ylabel(f"{met} (dB)",fontproperties=font)
                else:
                    plt.ylabel(f"{met}",fontproperties=font)
                #   plt.title(f" {dtset} dataset",loc = 'center',fontproperties=font)

                #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                #font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=12)
                font1 = {'family':'Times New Roman','style':'normal','size':self.legendsize}
                legend1 = plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                frame1 = legend1.get_frame()
                frame1.set_alpha(1)
                frame1.set_facecolor('none')  # 设置图例legend背景透明

                ax = plt.gca()#获得坐标轴的句柄
                ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                ax.spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                ax.spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                ax.spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细


                plt.tick_params(direction='in', axis='both',top=True,right=True, labelsize=16, width=3,)
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname('Times New Roman') for label in labels]
                [label.set_fontsize(20) for label in labels] #刻度值字号

                font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)

                plt.suptitle(f'AWGN channel, {dtset} dataset', fontproperties=font4,)

                plt.tight_layout(pad=0.5 , h_pad=1, w_pad=1)#  使得图像的四周边缘空白最小化
                #plt.subplots_adjust(top=0.92, bottom=0.01, left=0.01, right=0.99, wspace=0.2, hspace=0.2)#调节两个子图间的距离
                out_fig = plt.gcf()
                file = f"Test_{dtset}_{met}_Plot_SNR=({'-'.join([str(R) for R in self.SNR])})_R=({'-'.join([str(R) for R in self.R])}))"
                out_fig.savefig(self.getSavePath(file+".pdf"), bbox_inches = 'tight',pad_inches = 0.2)
                out_fig.savefig(self.getSavePath(file+".eps"), bbox_inches = 'tight',pad_inches = 0.2)
                out_fig.savefig(self.getSavePath(file+".png"), bbox_inches = 'tight',pad_inches = 0.2)
                plt.show()
                plt.close(fig)
        return

# choice =  2

# if choice == 1:
#     now = '2022-10-14-09:56:05'
# elif choice ==2:
now = '2022-10-12-17:38:12'


pl = ResultPlot(args, now) # 1: '2022-10-12-17:38:12'  2:'2022-10-14-09:56:05'
pl.ReadResult()



# pl.PlotTestMetricInOneFig()
# pl.PlotTestMetricSeperate()
# ##pl.plot_AllTrainMetric()
# pl.plot_AllTrainMetric1()




# pl.PlotTestMetricInOneFigArodic()
# pl.PlotTestMetricSeperateArodic()
# # ##pl.plot_AllTrainMetric()
# pl.plot_AllTrainMetricArodic()

#pl.PlotTestPSNR2RInOneFigArodic()
pl.PlotTestPSNR2RSeperateArodic()







