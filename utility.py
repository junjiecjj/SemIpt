
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""
import os,sys
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import torch.nn as nn
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import collections
from torch.utils.tensorboard import SummaryWriter
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
font = FontProperties(fname=fontpath+"SimSun.ttc", size = 22)#fname =  "/usr/share/fonts/truetype/arphic/SimSun.ttf",
font1 = FontProperties(fname=fontpath+"SimSun.ttc", size = 18)
font2 = FontProperties(fname=fontpath+"SimSun.ttc", size = 24)
font3 = FontProperties(fname=fontpath+"SimSun.ttc", size = 30)

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
fonte1 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 24)


fontt  = {'family':'Times New Roman','style':'normal','size':17}
fonttX  = {'family':'Times New Roman','style':'normal','size':22}
fonttY  = {'family':'Times New Roman','style':'normal','size':22}
fonttitle = {'style':'normal','size':17 }
fontt2 = {'style':'normal','size':19,'weight':'bold'}
fontt3  = {'style':'normal','size':16,}


def printArgs(args):
    print("############################################################################################")
    print("################################  args  ####################################################")
    print("############################################################################################")
    for k, v in args.__dict__.items():
        print(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}")
    print("################################  end  #####################################################")

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
        self.timer = 0


    # 从计时开始到现在的时间.
    def hold(self):
        return time.time() - self.t0

# https://developer.51cto.com/article/712616.html
from dataclasses import dataclass, field
import time
from typing import Callable, ClassVar, Dict, Optional
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(object):
    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time


# https://python3-cookbook.readthedocs.io/zh_CN/latest/c13/p13_making_stopwatch_timer.html
class Timer1:
    def __init__(self, func=time.perf_counter):
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

# 功能：
#
class checkpoint():
    def __init__(self, args ):
        self.args = args
        self.ok = True
        self.n_processes = 8
        self.mark = False
        self.startEpoch = 0   # 日志里各个压缩率和信噪比训练的epoch
        self.LastSumEpoch = 0 #日志里所有的压缩率和信噪比下训练的epoch之和
        self.SumEpoch = 0     # 本次训练的累计epoch

        # 模型训练时PSNR、MSE和loss和优化器等等数据的保存以及画图目录
        self.dir = os.path.join(args.save, f"TrainLog_{args.modelUse}")
        print(f"self.dir = {self.dir} \n")
        os.makedirs(self.dir, exist_ok=True)

        # 模型参数保存的目录
        self.modeldir = os.path.join(args.save, 'model')
        os.makedirs(self.modeldir, exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('trainLog.txt')) else 'w'
        self.log_file = open(self.get_path('trainLog.txt'), open_type)

        self.now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.writeArgsLog()

        self.metricLog = {}

        if os.path.isfile(self.get_path('TrainMetric_log.pt')):
            self.metricLog = torch.load(self.get_path('TrainMetric_log.pt'))
            epoch, sepoch = self.checkSameLen()
            self.LastSumEpoch = sepoch
            if self.mark == True:
                self.startEpoch = epoch
                print(f'\n从epoch={epoch}继续训练...\n' )
            else:
                print(f'\nepoch验证不通过, 重新开始训练...\n')

        if os.path.isfile(self.get_path('SumEpoch.pt')):
            self.SumEpoch = torch.load(self.get_path('SumEpoch.pt'))

        if args.reset:
            os.system('rm -rf ' + self.dir)

        print(color.fuchsia(f"\n#================================ checkpoint 准备完毕 =======================================\n"))

    # 更新全局的Epoch
    def UpdateEpoch(self):
        self.SumEpoch += 1
        return

    def writeArgsLog(self, open_type='a'):
        with open(self.get_path('argsConfig.txt'), open_type) as f:
            f.write('#==========================================================\n')
            f.write(self.now + '\n')
            f.write('#==========================================================\n\n')

            f.write("############################################################################################\n")
            f.write("################################  args  ####################################################\n")
            f.write("############################################################################################\n")

            for k, v in self.args.__dict__.items():
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}")
            f.write('\n')
            f.write("################################ args end  #################################################\n")

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


    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    # 保存模型参数
    def saveModel(self, trainer,  compratio, snr, epoch, is_best=False):
        trainer.model.save(self.modeldir, compratio, snr, epoch, is_best=is_best)
        return

    # 保存优化器参数
    def saveOptim(self, trainer):
        trainer.optimizer.save(self.dir)
        return

    # 画图和保存Loss日志
    def saveLoss(self, trainer):
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir)
        trainer.loss.plot_AllLoss(self.dir)
        return

    # 画图和保存PSNR等日志
    #@profile
    def save(self):
        self.plot_AllTrainMetric()
        torch.save(self.metricLog, self.get_path('TrainMetric_log.pt'))
        torch.save(self.SumEpoch, self.get_path('SumEpoch.pt'))
        return

    # 写日志
    def write_log(self, log, train=False ,refresh=True):
        # print(log)
        self.log_file.write(log + '\n')  # write() argument must be str, not dict
        if refresh:
            self.log_file.close()
            if train== True:
                self.log_file = open(self.get_path('trainLog.txt'), 'a')
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

        label = 'CompRatio={},SNR={}'.format(comprateTmp, snrTmp)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, self.metricLog[tmpS])
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)

        out_fig = plt.gcf()
        out_fig.savefig(self.get_path('train,epoch-psnr,CompRatio={},SNR={}.pdf'.format(comprateTmp, snrTmp)))
        plt.show()
        plt.close(fig)
        return

    #@profile
    def plot_AllTrainMetric(self):
        for idx, met in  enumerate(self.args.metrics):
            fig, axs=plt.subplots(len(self.args.SNRtrain),len(self.args.CompressRateTrain),figsize=(20,20))
            for comprate_idx, comprateTmp in enumerate(self.args.CompressRateTrain):
                for snr_idx, snrTmp in enumerate(self.args.SNRtrain):
                    tmpS = "MetricLog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
                    epoch = len(self.metricLog[tmpS])
                    X = np.linspace(1, epoch, epoch)

                    label = 'CompRatio={},SNR={},Metric={}'.format(comprateTmp, snrTmp,met)
                    axs[snr_idx,comprate_idx].set_title(label)

                    axs[snr_idx,comprate_idx].plot(X, self.metricLog[tmpS][:,idx],'r-',label=label,)
                    axs[snr_idx,comprate_idx].legend()
                    axs[snr_idx,comprate_idx].set_xlabel('Epochs')
                    axs[snr_idx,comprate_idx].set_ylabel(f"{met}")
                    axs[snr_idx,comprate_idx].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
            fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离
            plt.tight_layout()#  使得图像的四周边缘空白最小化
            out_fig = plt.gcf()
            out_fig.savefig(self.get_path(f"{met}_Epoch_Plot.pdf"))
            plt.show()
            plt.close(fig)
            gc.collect()
        return

# <<< 训练结果画图


# >>> 测试相关函数
    # 初始化测试结果目录
    def InittestDir(self, now = 'TestResult'):
        self.TeMetricLog = {}
        # now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.testRudir = os.path.join(self.dir, now)
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
                f.write(f"{k: <25}: {str(v): <40}  {str(type(v)): <20}")
            f.write('\n')
            f.write("################################ args end  #################################################\n")

        return

    def get_testpath(self, *subdir):
        return os.path.join(self.testRudir, *subdir)


# <<< 测试过程不同数据集上的的PSNR等指标随压缩率、信噪比的动态记录
    def InitTestMetric(self, comprateTmp, dataset):
        tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        if tmpS not in self.TeMetricLog.keys():
            self.TeMetricLog[tmpS] = torch.Tensor()
        else:
            pass
        return

    def AddTestMetric(self, comprateTmp, snrTmp, dataset):
        tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)

        # 第一列为snr, 后面各列为各个指标
        self.TeMetricLog[tmpS] = torch.cat([self.TeMetricLog[tmpS], torch.zeros(1, len(self.args.metrics)+1 )],dim=0)
        self.TeMetricLog[tmpS][-1,0]=snrTmp
        return

    def UpdateTestMetric(self, comprateTmp, dataset, metric):
        tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        self.TeMetricLog[tmpS][-1,1:] += metric

    def MeanTestMetric(self, comprateTmp, dataset, n_images):
        tmpS = "TestMetricLog:Dataset={},CompRatio={}".format(dataset,comprateTmp)
        self.TeMetricLog[tmpS][-1,1:] /= n_images
        return self.TeMetricLog[tmpS][-1,1:]
# 训练过程的PSNR等指标的动态记录 >>>

    def SaveTestLog(self):
        self.plot_AllTestMetric()
        torch.save(self.TeMetricLog, self.get_testpath('TestMetric_log.pt'))
        return


    def plot_AllTestMetric(self):
        for dsIdx,dtset in enumerate(self.args.data_test):
            fig, axs=plt.subplots(len(self.args.CompressRateTrain),len(self.args.metrics),figsize=(20,20))
            for crIdx, compratio in enumerate(self.args.CompressRateTrain):
                for metIdx, met in enumerate(self.args.metrics):
                    label = f"CompressRate={compratio}"
                    tmps = "TestMetricLog:Dataset={},CompRatio={}".format(dtset,compratio)
                    data = self.TeMetricLog[tmps]
                    axs[crIdx,metIdx].set_title(label,loc = 'left',fontdict=fonttitle)
                    axs[crIdx,metIdx].plot(data[:,0], data[:,metIdx+1],'r-',label=label,)
                    axs[crIdx,metIdx].set_xlabel('SNR',fontdict=fonttX)
                    axs[crIdx,metIdx].set_ylabel(f"{met}",fontdict=fonttY)
                    fonte = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 16)
                    axs[crIdx,metIdx].legend(borderaxespad=0,edgecolor='black',prop=fonte,)
                    axs[crIdx,metIdx].spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
                    axs[crIdx,metIdx].spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
                    axs[crIdx,metIdx].spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
                    axs[crIdx,metIdx].spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

                    fontt2 = {'family':'Times New Roman','style':'normal','size':16}
                    legend1 = axs[crIdx,metIdx].legend(loc='best',borderaxespad=0,edgecolor='black',prop=fontt2,)
                    frame1 = legend1.get_frame()
                    frame1.set_alpha(1)
                    frame1.set_facecolor('none') # 设置图例legend背景透明

                    axs[crIdx,metIdx].tick_params(labelsize=16,width=3)
                    labels = axs[crIdx,metIdx].get_xticklabels() + axs[crIdx,metIdx].get_yticklabels()
                    [label.set_fontname('Times New Roman') for label in labels]
                    [label.set_fontsize(20) for label in labels] #刻度值字号

            fig.subplots_adjust(hspace=0.25)#调节两个子图间的距离
            font4 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
            plt.suptitle(f"{self.args.metrics[0]}, {self.args.metrics[1]} metric of {dtset}",x=0.5,y=0.93,fontproperties=font4,)
            #plt.tight_layout()#  使得图像的四周边缘空白最小化

            out_fig = plt.gcf()
            out_fig.savefig(self.get_testpath(f"{dtset}TestMetrics_Plot.pdf"), bbox_inches = 'tight',pad_inches = 0.2)
            plt.show()
            plt.close(fig)
        return
    def SaveTestFig(self, DaSetName, CompRatio, Snr, figname, data):
        filename = self.get_testpath('results-{}'.format(DaSetName),'{}_CompRa={}_Snr={}.png'.format(figname, CompRatio,Snr))
        print(f"filename = {filename}\n")
        normalized = data[0].mul(255 / self.args.rgb_range)
        tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
        print(f"tensor_cpu.shape = {tensor_cpu.shape}\n")
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
            filename = self.get_path('results-{}'.format(dataset.dataset.name),'{}_x{}_'.format(filename, scale))

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))
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

# ckp.SaveTestLog()




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


def calc_psnr(sr, hr, scale, rgb_range, cal_type='y'):
    if hr.nelement() == 1: return 0

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


def make_optimizer(args, net):
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
            kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}  # args.gamma =0.5
            self.scheduler = scheduler_class(self, **kwargs_scheduler)

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer



# model = net()
# LR = 0.01
# opt = make_optimizer(args,model)
# loss = torch.nn.CrossEntropyLoss()

# lr_list1 = []
# lr_list2 = []
# for epoch in range(200):
#      for i in range(20):
#          y = torch.randint(0, 9, (10,10))*1.0
#          opt.zero_grad()
#          out = model(torch.randn(10,1))
#          lss = loss(out, y)
#          lss.backward()
#          opt.step()
#      opt.schedule()
#      lr_list2.append(opt.get_lr())
#      lr_list1.append(opt.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(200),lr_list1,color = 'r')
# #plt.plot(range(100),lr_list2,color = 'b')
# out_fig = plt.gcf()
# plt.show()



#  使用时：
"""

model = net()
LR = 0.01
optimizer = make_optimizer( args,  model)


lr_list1 = []

for epoch in range(200):
    for X,y in dataloder:
        optimizer.step()
    optimizer.schedule()
    lr_list1.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(200),lr_list1,color = 'r')

plt.show()

"""