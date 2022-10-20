
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



def printArgs(args):
    print("############################################################################################")
    print("################################  args  ####################################################")
    print("############################################################################################")
    for k, v in args.__dict__.items():
        print(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}")
    print("################################  end  #####################################################")

# 初始化随机数种子
def set_random_seed(seed = 10,deterministic=False,benchmark=False):
    random.seed(seed)
    np.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True

# Timer
class timer(object):
    def __init__(self,name='epoch'):
        self.acc = 0
        self.name = name
        self.timer = 0
        self.tic()

    def tic(self):  # time.time()函数返回自纪元以来经过的秒数。
        self.t0 = time.time()
        self.ts = self.t0

    # 返回从ts开始历经的秒数。
    def toc(self):
        diff = time.time() - self.ts
        self.ts = time.time()
        self.timer  += diff
        return diff

    def reset(self):
        self.ts = time.time()
        tmp = self.timer
        self.timer = 0
        return tmp

    # 从计时开始到现在的时间.
    def hold(self):
        return time.time() - self.t0



# 功能：
class checkpoint():
    def __init__(self, args ):
        self.args = args
        self.ok = True
        self.n_processes = 8
        self.mark = False
        self.startEpoch = 0     # 日志里各个压缩率和信噪比训练的epoch
        self.LastSumEpoch = 0   # 日志里所有的压缩率和信噪比下训练的epoch之和
        self.SumEpoch = 0       # 本次训练的累计epoch

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # 模型训练时PSNR、MSE和loss和优化器等等数据的保存以及画图目录
        self.savedir = os.path.join(args.save, f"{self.now}_TrainLog_{args.modelUse}")
        self.loaddir = os.path.join(args.load+f"_TrainLog_{args.modelUse}")
        print(f"训练过程MSR、PSNR、Loss等保存目录 = {self.savedir} \n")
        #if args.reset:
            #print(f"删除目录:{self.dir}")
            #os.system('rm -rf ' + self.dir)
        os.makedirs(self.savedir, exist_ok=True)

        # 模型参数保存的目录
        self.modelSaveDir = os.path.join(args.saveModel, f"{self.now}_Model_{args.modelUse}")
        self.modelLoadDir = os.path.join(args.loadModel+f"_Model_{args.modelUse}")

        if args.reset:
            print(f"删除目录:{self.modelSaveDir}")
            os.system('rm -rf ' + self.modelSaveDir)

        os.makedirs(self.modelSaveDir, exist_ok=True)

        open_type = 'a' if os.path.exists(self.getSavePath('trainLog.txt')) else 'w'
        self.log_file = open(self.getSavePath('trainLog.txt'), open_type)

        self.writeArgsLog()

        self.metricLog = {}

        if os.path.isfile(self.getLoadPath('TrainMetric_log.pt')):
            self.metricLog = torch.load(self.getLoadPath('TrainMetric_log.pt'))
            epoch, sepoch = self.checkSameLen()
            self.LastSumEpoch = sepoch
            if self.mark == True:
                self.startEpoch = epoch
                print(f'\n从epoch={epoch}继续训练...\n' )
            else:
                print(f'\nepoch验证不通过, 重新开始训练...\n')

        if os.path.isfile(self.getLoadPath('SumEpoch.pt')):
            self.SumEpoch = torch.load(self.getLoadPath('SumEpoch.pt'))

        print(color.fuchsia(f"\n#================================ checkpoint 准备完毕 =======================================\n"))

    # 更新全局的Epoch
    def UpdateEpoch(self):
        self.SumEpoch += 1
        return

    def writeArgsLog(self, open_type='a'):
        with open(self.getSavePath('argsConfig.txt'), open_type) as f:
            f.write('#====================================================================================\n')
            f.write(self.now + '\n')
            f.write('#====================================================================================\n\n')

            f.write("###############################################################################\n")
            f.write("################################  args  #######################################\n")
            f.write("###############################################################################\n")

            for k, v in self.args.__dict__.items():
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}\n")
            f.write("\n################################ args end  ##################################\n")
        return



    # 因为多个不同压缩率的不同层是融合在一个模型里的，所以需要检查在每个压缩率和信噪比下训练的epoch是否相等
    def checkSameLen(self):
        lens = []
        sumepoch = 0
        for key in list(self.metricLog.keys()):
            lens.append(len(self.metricLog[key]))
            sumepoch +=  len(self.metricLog[key])
        set1 = set(lens)
        if lens == []:
            print(f"Epoch == 0, 重新训练.....\n")
        elif len(set1) == 1 and lens[0]>=1:
            #print(f"所有的压缩率和信噪比组合都训练了等长的Epoch...\n")
            self.mark = True
            return lens[0], sumepoch
        else:
            print(f"所有的压缩率和信噪比组合下的Epoch不等...\n")
            self.mark = False
            return 0, sumepoch


# <<< 训练过程的PSNR等指标的动态记录
    def InitMetricLog(self, comprateTmp, snrTmp):
        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
        if tmpS not in self.metricLog.keys():
            self.metricLog[tmpS] = torch.Tensor()
        else:
            pass
        return

    #@profile
    def AddMetricLog(self, comprateTmp, snrTmp):
        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)

        self.metricLog[tmpS] = torch.cat([ self.metricLog[tmpS], torch.zeros(1, len(self.args.metrics))])
        return

    def UpdateMetricLog(self, comprateTmp, snrTmp, metric):
        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
        self.metricLog[tmpS][-1] += metric
        return

    def MeanMetricLog(self, comprateTmp, snrTmp, n_batch):
        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
        self.metricLog[tmpS][-1] /= n_batch
        return self.metricLog[tmpS][-1]
# 训练过程的PSNR等指标的动态记录 >>>


    def getSavePath(self, *subdir):
        return os.path.join(self.savedir, *subdir)

    def getLoadPath(self, *subdir):
        return os.path.join(self.loaddir, *subdir)

    # 保存模型参数
    def saveModel(self, trainer,  compratio, snr, epoch, is_best=False):
        trainer.model.save(self.modelSaveDir, compratio, snr, epoch, is_best=is_best)
        return

    # 保存优化器参数
    def saveOptim(self, trainer):
        trainer.optimizer.save(self.savedir)
        return

    # 画图和保存Loss日志
    def saveLoss(self, trainer):
        trainer.loss.save(self.savedir)
        trainer.loss.plot_loss(self.savedir)
        trainer.loss.plot_AllLoss(self.savedir)
        return

    # 画图和保存PSNR等日志
    #@profile
    def save(self):
        self.plot_AllTrainMetric()
        torch.save(self.metricLog, self.getSavePath('TrainMetric_log.pt'))
        torch.save(self.SumEpoch, self.getSavePath('SumEpoch.pt'))
        return

    # 写日志
    def write_log(self, log, train=False ,refresh=True):
        # print(log)
        self.log_file.write(log + '\n')  # write() argument must be str, not dict
        if refresh:
            self.log_file.close()
            if train== True:
                self.log_file = open(self.getSavePath('trainLog.txt'), 'a')
            else:
                self.log_file = open(self.get_testpath('testLog.txt'), 'a')
        return

    # 关闭日志
    def done(self):
        self.log_file.close()
        return

# >>> 训练结果画图
    def plot_trainPsnr(self, comprateTmp, snrTmp):
        tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)

        epoch = len(self.metricLog[tmpS])

        axis = np.linspace(1, epoch, epoch)

        label = 'CompRatio={},SNR={}(dB)'.format(comprateTmp, snrTmp)
        fig = plt.figure(constrained_layout=True)
        plt.title(label)
        plt.plot(axis, self.metricLog[tmpS])
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)

        out_fig = plt.gcf()
        out_fig.savefig(self.getSavePath('train,epoch-psnr,CompRatio={},SNR={}.pdf'.format(comprateTmp, snrTmp)))
        plt.show()
        plt.close(fig)
        return


    """
    两张大图；每张图对应一个指标，PSNR或者MSE，下面以PSNR为例；
    每张图有len(self.args.SNRtrain)xlen(self.args.CompressRateTrain)个子图；
    每个子图对应在指定压缩率和信噪比下训练时PSNR随着epoch变化曲线；
    """
    #@profile
    def plot_AllTrainMetric(self):

        width = 6
        high = 5
        figWidth =width*len(self.args.CompressRateTrain)
        figHigh = high*len(self.args.SNRtrain)

        for idx, met in  enumerate(self.args.metrics):

            # 如果信噪比和压缩率只有一个
            if len(self.args.SNRtrain) == 1 and len(self.args.CompressRateTrain) == 1:
                fig = plt.figure()# constrained_layout=True
                tmpS = "MetricLog:CompRatio={},SNR={}".format(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
                #label = 'R={},SNR={}(dB)'.format(self.args.CompressRateTrain[0], self.args.SNRtrain[0])
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
                #plt.title(label, fontproperties=font)

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
                        font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=16)
                        font1 = {'family':'Times New Roman','style':'normal','size':16}
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

            #调节两个子图间的距离
            # plt.subplots_adjust(top=0.92,bottom=0.1, left=0.1, right=0.97, wspace=0.4, hspace=0.2)
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(self.getSavePath(f"Train_{met}_EachRandSNREpoch_Plot.pdf"))
            out_fig.savefig(self.getSavePath(f"Train_{met}_EachRandSNREpoch_Plot.eps"))
            plt.show()
            plt.close(fig)
        return


# <<< 训练结果画图


# >>> 测试相关函数
    # 初始化测试结果目录
    def InittestDir(self, now = 'TestResult'):
        self.TeMetricLog = {}
        # now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.testRudir = os.path.join(self.savedir, now)
        os.makedirs(self.testRudir,exist_ok=True)
        for d in self.args.data_test:
            os.makedirs(os.path.join(self.testRudir,'results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_testpath('testLog.txt')) else 'w'
        self.log_file = open(self.get_testpath('testLog.txt'), open_type)
        print(f"====================== 打开测试日志 {self.log_file.name} ===================================")

        with open(self.get_testpath('argsTest.txt'), open_type) as f:
            f.write('#==========================================================\n')
            f.write(self.now + '\n')
            f.write('#==========================================================\n\n')

            f.write("############################################################################################\n")
            f.write("####################################  Test args  ###########################################\n")
            f.write("############################################################################################\n")

            for k, v in self.args.__dict__.items():
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}\n")
            f.write('\n')
            f.write("################################ args end  #################################################\n")
        return



    def get_testpath(self, *subdir):
        return os.path.join(self.testRudir, *subdir)

# <<< 测试过程不同数据集上的的PSNR等指标随压缩率、信噪比的动态记录
    def InitTestMetric(self, comprateTmp, dataset):
        #tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dataset,comprateTmp,self.args.SNRtrain[0])
        if tmpS not in self.TeMetricLog.keys():
            self.TeMetricLog[tmpS] = torch.Tensor()
        else:
            pass
        return

    def AddTestMetric(self, comprateTmp, snrTmp, dataset):
        #tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dataset,comprateTmp,self.args.SNRtrain[0])
        # 第一列为snr, 后面各列为各个指标
        self.TeMetricLog[tmpS] = torch.cat([self.TeMetricLog[tmpS], torch.zeros(1, len(self.args.metrics)+1 )],dim=0)
        self.TeMetricLog[tmpS][-1,0]=snrTmp
        return

    def UpdateTestMetric(self, comprateTmp, dataset, metric):
        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dataset,comprateTmp,self.args.SNRtrain[0])

        self.TeMetricLog[tmpS][-1,1:] += metric

    def MeanTestMetric(self, comprateTmp, dataset, n_images):
        # tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dataset,comprateTmp,self.args.SNRtrain[0])
        self.TeMetricLog[tmpS][-1,1:] /= n_images
        return self.TeMetricLog[tmpS][-1,1:]
# 训练过程的PSNR等指标的动态记录 >>>

    def SaveTestLog(self):
        # self.plot_AllTestMetric()
        self.PlotTestMetricSeperate()
        self.PlotTestMetricInOneFig()
        torch.save(self.TeMetricLog, self.get_testpath('TestMetric_log.pt'))
        return


    def plot_AllTestMetric(self):
        for dsIdx,dtset in enumerate(self.args.data_test):
            fig, axs=plt.subplots(len(self.args.CompressRateTrain),len(self.args.metrics),figsize=(20,20))
            for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                for metIdx, met in enumerate(self.args.metrics):
                    label = f"CompressRate={compratio}, {met}"
                    tmps = "TestMetricLog:Dataset={},CompRatio={}".format(dtset,compratio)
                    data = self.TeMetricLog[tmps]

                    axs[crIdx,metIdx].plot(data[:,0], data[:,metIdx+1],'r-',label=label,)
                    font = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 20)
                    axs[crIdx,metIdx].set_xlabel('SNR (dB)',fontproperties=font)
                    axs[crIdx,metIdx].set_ylabel(f"{met}",fontproperties=font)
                    axs[crIdx,metIdx].set_title(label,loc = 'left',fontproperties=font)

                    #font1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
                    font1 = FontProperties(fname=fontpath2+"Caskaydia Cove ExtraLight Nerd Font Complete.otf", size=20)
                    legend1 = axs[crIdx,metIdx].legend(loc='best', borderaxespad=0, edgecolor='black', prop=font1,)
                    frame1 = legend1.get_frame()
                    frame1.set_alpha(1)
                    frame1.set_facecolor('none')  # 设置图例legend背景透明

                    axs[crIdx,metIdx].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                    axs[crIdx,metIdx].spines['left'].set_linewidth(2);  ###设置左边坐标轴的粗细
                    axs[crIdx,metIdx].spines['right'].set_linewidth(2); ###设置右边坐标轴的粗细
                    axs[crIdx,metIdx].spines['top'].set_linewidth(2);   ###设置上部坐标轴的粗细

                    fontt2 = {'family':'Times New Roman','style':'normal','size':16}
                    legend1 = axs[crIdx,metIdx].legend(loc='best',borderaxespad=0,edgecolor='black',prop=fontt2,)
                    frame1 = legend1.get_frame()
                    frame1.set_alpha(1)
                    frame1.set_facecolor('none') # 设置图例legend背景透明

                    axs[crIdx,metIdx].tick_params(labelsize=16,width=3)
                    labels = axs[crIdx,metIdx].get_xticklabels() + axs[crIdx,metIdx].get_yticklabels()
                    [label.set_fontname('Times New Roman') for label in labels]
                    [label.set_fontsize(20) for label in labels] #刻度值字号

            fig.subplots_adjust(hspace=0.2)#调节两个子图间的距离
            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            plt.suptitle(f"Metric of {dtset}",x=0.5,y=0.93,fontproperties=font4,)
            #plt.tight_layout()#  使得图像的四周边缘空白最小化

            out_fig = plt.gcf()
            out_fig.savefig(self.get_testpath(f"{dtset}TestMetrics_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return

    """
    功能是：
    每张图对应一个数据集在不同压缩率下的PSNR/MSE-epoch曲线；
    每个数据集有两张图，每个图对应一个指标；
    对每张图有多条曲线，每一条曲线对应一个压缩率下的PSNR或MSE随SNR的变化曲线；
    """
    def PlotTestMetricSeperate(self):
        mark  = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']
        color = ['#808000','#C000C0', '#000000','#00FFFF','#0000FF', '##FF1493', '#ADFF2F','#FF8C00','#00FF00', '##800080', '#FF0000','#1E90FF']

        high = 5
        width = 6

        for dsIdx, dtset in enumerate(self.args.data_test):
            for metIdx, met in enumerate(self.args.metrics):
                # fig, axs = plt.subplots(raw, col, figsize=(col*Len, raw*Len),constrained_layout=True)
                fig = plt.figure(figsize=(width, high),)# constrained_layout=True
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    #tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dtset, compratio)
                    tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio, self.args.SNRtrain[0])
                    data = self.TeMetricLog[tmpS]
                    # lb = f"R={compratio}"
                    lb = r"$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, self.args.SNRtrain[0])
                    plt.plot(data[:,0], data[:,metIdx+1], linestyle='-',color=color[crIdx],marker=mark[crIdx], label = lb)

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
                out_fig.savefig(self.get_testpath(f"Test_{dtset}_{met}_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
                out_fig.savefig(self.get_testpath(f"Test_{dtset}_{met}_Plot.eps"), bbox_inches = 'tight',pad_inches = 0.2)
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
                for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                    tmpS = "TestMetricLog:Dataset={},CompRatio={},SNRtrain={}".format(dtset, compratio,self.args.SNRtrain[0])
                    data = self.TeMetricLog[tmpS]
                    lb = r"$\mathrm{R}_\mathrm{train}=%.2f,\mathrm{SNR}_\mathrm{train}=%d\mathrm{(dB)}$"%(compratio, self.args.SNRtrain[0])
                    axs[i,j].plot(data[:,0], data[:,metIdx+1], linestyle='-', color=color[crIdx], marker=mark[crIdx], label = lb)

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
            out_fig.savefig(self.get_testpath(f"Test_{met}_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            out_fig.savefig(self.get_testpath(f"Test_{met}_Plot.eps"), bbox_inches = 'tight',pad_inches = 0.2)
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

    def SaveTestFig(self, DaSetName, CompRatio, SnrTest, snrTrain, figname, data):
        filename = self.get_testpath('results-{}'.format(DaSetName),'{}_R={}_SnrTrain={}_SnrTest={}.png'.format(figname, CompRatio,snrTrain,SnrTest))
        #print(f"filename = {filename}\n")
        normalized = data[0].mul(255 / self.args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        #print(f"tensor_cpu.shape = {tensor_cpu.shape}\n")
        imageio.imwrite(filename, tensor_cpu.numpy())
        return

# <<< 测试相关函数

    def begin_queue(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())

        self.process = [ Process(target=bg_target, args=(self.queue,)) for _ in range(self.n_processes) ]

        for p in self.process:
            p.start()
        return

    def end_queue(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()
        return

    def save_results_byQueue(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.getSavePath('results-{}'.format(dataset.dataset.name),'{}_x{}_'.format(filename, scale))

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
        return


#  功能：将img每个像素点的至夹在[0,255]之间
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)



def calc_metric(sr, hr, scale, rgb_range, metrics, cal_type='y'):

    metric = []

    for met in metrics:
        if met == 'PSNR':
            psnr = calc_psnr(sr, hr, scale, rgb_range, cal_type='y')
        elif met == 'MSE':
            mse = calc_mse(sr, hr, scale)
        else:
            m = 0
    metric.append(psnr)
    metric.append(mse)
    return torch.tensor(metric)


"""
逐像素的计算均方误差
"""
def calc_psnr(sr, hr, scale, rgb_range, cal_type='y'):
    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range

    if cal_type=='y':
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    if scale == 1:
        valid = diff
    else:
        valid = diff[..., scale:-scale, scale:-scale]
    mse = valid.pow(2).mean()
    if mse <= 1e-10:
        mse = 1e-10
    return   -10 * math.log10(mse)

"""
逐像素的计算均方误差
"""
def calc_mse(sr, hr, scale):
    if hr.nelement() == 1: return 0

    diff = (sr - hr)

    if scale == 1:
        valid = diff
    else:
        valid = diff[..., scale:-scale, scale:-scale]
    mse = valid.pow(2).mean()
    return mse


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)


def make_optimizer(args, net, total_steps):
    '''
    make optimizer and scheduler together
    '''
    # optimizer
    #  filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，如果要转换为列表，可以使用 list() 来转换。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    trainable = filter(lambda x: x.requires_grad, net.parameters())
    #trainable = net.parameters()
    #  lr = 1e-4, weight_decay = 0
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    # optimizer = ADAM
    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas  # (0.9, 0.999)
        kwargs_optimizer['eps'] = args.epsilon  # 1e-8
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler, milestones = 0,   gamma = 0.5
    milestones = list(map(lambda x: int(x), args.decay.split('-')))  #  [20, 40, 60, 80, 100, 120]
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}  # args.gamma =0.5
    scheduler_class = lrs.MultiStepLR

    warmup_class = optimization.get_polynomial_decay_schedule_with_warmup
    kwargs_warmup = {"num_warmup_steps":args.warm_up_ratio*total_steps, "num_training_steps":total_steps,"power":args.power,"lr_end":args.lr_end}

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            if os.path.isfile(self.get_dir(load_dir)):
                self.load_state_dict(torch.load(self.get_dir(load_dir)))
                if epoch > 1:
                    print(f"加载优化器参数.....")
                    for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

        def reset_state(self):
            self.state = collections.defaultdict(dict)
            #self.scheduler.last_epoch = 0
            #self.scheduler._last_lr = 0
            for param_group in self.param_groups:
                param_group["lr"] = args.lr

            milestones = list(map(lambda x: int(x), args.decay.split('-')))  #  [20, 40, 60, 80, 100, 120]
            # kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}  # args.gamma =0.5
            # self.scheduler = scheduler_class(self, **kwargs_scheduler)

            kwargs_warmup = {"num_warmup_steps":args.warm_up_ratio*total_steps, "num_training_steps":total_steps,"power":args.power,"lr_end":args.lr_end}
            self.scheduler = warmup_class(self, **kwargs_warmup)

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    #optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    optimizer._register_scheduler(warmup_class, **kwargs_warmup)

    return optimizer



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
