
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""
import os
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


# 本项目自己编写的库
from option import args

# Timer
class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):  # time.time()函数返回自纪元以来经过的秒数。
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0

# 功能：
#
class checkpoint():
    def __init__(self, args ):
        self.args = args
        self.ok = True
        self.n_processes = 8
        self.mark = False

        self.dir = args.save
        print(f"self.dir = {self.dir} \n")
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(args.save, 'model'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('trainLog.txt')) else 'w'
        self.log_file = open(self.get_path('trainLog.txt'), open_type)

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        with open(self.get_path('argsConfig.txt'), open_type) as f:
            f.write('#==========================================================\n')
            f.write(now + '\n')
            f.write('#==========================================================\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.psnrlog = {}

        if os.path.isfile(self.get_path('trainPsnr_log.pt')):
            self.psnrlog = torch.load(self.get_path('trainPsnr_log.pt'))
            epoch = self.checkSameLen()
            if self.mark == True:
                print('\n从epoch={}继续训练...\n'.format(len(self.psnrlog['psnrlog:CompRatio=0.17,SNR=6'])))
            else:
                print('\nepoch验证不通过，重新开始训练...\n')

        if args.reset:
            os.system('rm -rf ' + self.dir)

    def checkSameLen(self):
        lens = []
        for key in list(self.psnrlog.keys()):
            lens.append(len(self.psnrlog[key]))
        lens = set(lens)
        if len(lens) == 1:
            print(f"所有的压缩率和信噪比组合都训练了等长的Epoch...\n")
            self.mark = True
            return
        else:
            print(f"所有的压缩率和信噪比组合下的Epoch不等...\n")
            self.mark = False


    def InitPsnrLog(self, comprateTmp, snrTmp):
        tmpS = "psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
        if tmpS not in self.psnrlog.keys():
            self.psnrlog[tmpS] = torch.Tensor()
        else:
            pass

    def AddPsnrLog(self, comprateTmp, snrTmp):
        tmpS = "psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)

        self.psnrlog[tmpS] = torch.cat([ self.psnrlog[tmpS], torch.zeros(1, 1)])

    def UpdatePsnrLog(self, comprateTmp, snrTmp, psnr):
        tmpS = "psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
        self.psnrlog[tmpS][-1] += psnr

    def meanPsnrLog(self, comprateTmp, snrTmp, n_batch):
        tmpS = "psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
        self.psnrlog[tmpS][-1] /= n_batch


    def InittestDir(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.testResDir = os.path.join(self.dir, now)
        os.makedirs(self.testResDir)
        for d in self.args.data_test:
            os.makedirs(os.path.join(self.testResDir,'results-{}'.format(d)), exist_ok=True)


    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        #trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        #trainer.loss.save(self.dir)
        #trainer.loss.plot_loss(self.dir, epoch)

        self.plot_AllTrainPsnr( )
        # trainer.optimizer.save(self.dir)
        torch.save(self.psnrlog, self.get_path('trainPsnr_log.pt'))



    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')  # write() argument must be str, not dict
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('trainLog.txt'), 'a')

    def done(self):
        self.log_file.close()


    def plot_trainPsnr(self, comprateTmp, snrTmp):
        tmpS = "psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)

        epoch = len(self.psnrlog[tmpS])

        axis = np.linspace(1, epoch, epoch)

        label = 'CompRatio={},SNR={}'.format(comprateTmp, snrTmp)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, self.psnrlog[tmpS])
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)

        out_fig = plt.gcf()
        out_fig.savefig(self.get_path('train,epoch-psnr,CompRatio={},SNR={}.pdf'.format(comprateTmp, snrTmp)))
        plt.show()
        plt.close(fig)



    def plot_AllTrainPsnr(self):
        fig, axs=plt.subplots(len(self.args.SNRtrain),len(self.args.CompressRateTrain),figsize=(20,20))
        for comprate_idx, comprateTmp in enumerate(self.args.CompressRateTrain):
            for snr_idx, snrTmp in enumerate(self.args.SNRtrain):
                tmpS = "psnrlog:CompRatio={},SNR={}".format(comprateTmp, snrTmp)
                epoch = len(self.psnrlog[tmpS])
                axis = np.linspace(1, epoch, epoch)

                label = 'CompRatio={},SNR={}'.format(comprateTmp, snrTmp)
                axs[snr_idx,comprate_idx].set_title(label)

                axs[snr_idx,comprate_idx].plot(axis, self.psnrlog[tmpS],'r-',label=label,)
                axs[snr_idx,comprate_idx].legend()
                axs[snr_idx,comprate_idx].set_xlabel('Epochs')
                axs[snr_idx,comprate_idx].set_ylabel('PSNR')
                #axs[snr_idx,comprate_idx].grid(True)
                #axs[snr_idx,comprate_idx].tick_params(direction='in')
                axs[snr_idx,comprate_idx].tick_params(direction='in',axis='both',top=True,right=True,labelsize=16,width=3)
        fig.subplots_adjust(hspace=0.6)#调节两个子图间的距离
        plt.tight_layout()#  使得图像的四周边缘空白最小化
        out_fig = plt.gcf()
        out_fig.savefig(self.get_path('AllPsnr.pdf'))
        plt.show()
        plt.close(fig)



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

    def end_queue(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results_byQueue(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path('results-{}'.format(dataset.dataset.name),'{}_x{}_'.format(filename, scale))

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


ckp = checkpoint(args)
# 依次遍历压缩率
for comprate_idx, compressrate in enumerate(args.CompressRateTrain):  #[0.17, 0.33, 0.4]
    # 依次遍历信噪比
    for snr_idx, snr in enumerate(args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
        #print(f"\nNow， Train on comprate_idx = {comprate_idx}, compressrate = {compressrate}， snr_idx = {snr_idx}, snr = {snr}, \n")

        epoch = 0

        ckp.InitPsnrLog(compressrate, snr)
        # 遍历epoch
        for epoch_idx in  range(10):
            epoch += 1
            #初始化特定信噪比和压缩率下的存储字典
            ckp.AddPsnrLog(compressrate, snr)

            # 遍历训练数据集
            for i in range(20):
                # pass
                ckp.UpdatePsnrLog(compressrate, snr, epoch_idx+i)
            ckp.meanPsnrLog(compressrate, snr, 20)

ckp.plot_trainPsnr(0.4, 18)
#ckp.plot_AllTrainPsnr()



#  功能：将img每个像素点的至夹在[0,255]之间
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

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

    return -10 * math.log10(mse)


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