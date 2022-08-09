
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

如果类B的某个成员self.fa是类A的实例a, 则如果B中更改了a的某个属性, 则a的那个属性也会变.

"""


import sys,os
import utility
import torch
from torch.autograd import Variable
from tqdm import tqdm
import datetime
import torch.nn as nn
import imageio



# 本项目自己编写的库
from ColorPrint  import ColoPrint
color = ColoPrint()
# print(color.fuchsia("Color Print Test Pass"))

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, writer):
        self.args = args
        self.scale = args.scale
        #print(f"trainer  self.scale = {self.scale} \n")
        self.wr = writer
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        #self.wr.WrModel(self.model.model, torch.randn(16, 3, 48, 48))
        if self.args.load != '':
            if self.ckp.mark == True:
                self.optimizer.load(self.ckp.dir)

        self.error_last = 1e8


    def prepare(self, *args):
        #device = torch.device('cpu' if self.args.cpu else 'cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


    def train(self):
        #lossFn = nn.MSELoss()
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.fuchsia(f"\n#================================ 开始训练, 时刻:{now} =======================================\n"))

        print(f"#======================================== 训练过程 =============================================",  file=self.ckp.log_file)

        self.model.train()
        torch.set_grad_enabled(True)
        #ind1_scale = self.args.scale.index(1)


        tm = utility.timer()

        #self.loader_train.dataset.set_scale(ind1_scale)
        #print(f"scale in train = {self.loader_train.dataset.scale[self.loader_train.dataset.idx_scale]}\n")
        accumEpoch = 0
        # 依次遍历压缩率
        for comprate_idx, compressrate in enumerate(self.args.CompressRateTrain):  #[0.17, 0.33, 0.4]
            # 依次遍历信噪比
            for snr_idx, snr in enumerate(self.args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
                print(color.fuchsia( f" 开始在压缩率索引为:{comprate_idx}, 压缩率为:{compressrate}, 信噪比索引为:{snr_idx}, 信噪比为:{snr} 下训练\n"))
                self.ckp.write_log(f"开始在压缩率索引为:{comprate_idx}, 压缩率为:{compressrate}, 信噪比索引为:{snr_idx}, 信噪比为:{snr} 下训练", train=True)
                # 初始化 特定信噪比和压缩率下 的Psnr日志
                self.ckp.InitMetricLog(compressrate, snr)

                # 遍历epoch
                for epoch_idx in  range(self.ckp.startEpoch, self.ckp.startEpoch+self.args.epochs):

                    accumEpoch += 1
                    self.ckp.UpdateEpoch()
                    #print(f"ckp.SumEpoch = {self.ckp.SumEpoch.requires_grad}\n")

                    #初始化loss日志
                    self.loss.start_log()

                    # 动态增加特定信噪比和压缩率下的Psnr等评价指标日志
                    self.ckp.AddMetricLog(compressrate, snr)

                    loss = 0
                    # 遍历训练数据集
                    for batch_idx, (lr, hr, filename)  in enumerate(self.loader_train):
                        #print(f"{batch_idx}, lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}\n")
                        # lr.shape = torch.Size([32, 3, 48, 48]), hr.shape = torch.Size([32, 3, 48, 48]), filename = ('0052', '0031',
                        lr, hr = self.prepare(lr, hr)
                        #print(f"lr.dtype = {lr.dtype}, hr.dtype = {hr.dtype}") #lr.dtype = torch.float32, hr.dtype = torch.float32
                        #hr = hr.to(device)
                        #lr = lr.to(device)
                        print(f"lr.requires_grad = {lr.requires_grad}, hr.requires_grad = {hr.requires_grad} \n")
                        sr = self.model(hr, idx_scale=0, snr=snr, compr_idx=comprate_idx)

                        # 计算batch内的loss
                        lss = self.loss(sr, hr)
                        #print(f"lss = {lss}")
                        # lss.requires_grad_(True)
                        # lss = Variable(lss, requires_grad = True)
                        #print(f"lss.grad_fn = {lss.grad_fn}\n")
                        #print(f"lss.requires_grad = {lss.requires_grad}\n") #lss.requires_grad = True

                        self.optimizer.zero_grad() # 必须在反向传播前先清零。
                        lss.backward()
                        self.optimizer.step()
                        #optimizer.step()

                        # 计算bach内的psnr和MSE
                        # with torch.no_grad():
                        metric = utility.calc_metric(sr=sr, hr=hr, scale=1, rgb_range=self.args.rgb_range, metrics=self.args.metrics)
                        # print(f"metric.requires_grad = {metric.requires_grad} {metric.dtype}")

                        # 更新 bach内的psnr
                        self.ckp.UpdateMetricLog(compressrate, snr, metric)

                        self.ckp.write_log(f"\t\t训练完一个 batch: loss = {lss}, metric = {metric} \n", train=True)
                        print(f"\t\tEpoch {epoch_idx}/{self.ckp.startEpoch+self.args.epochs}, Iter {batch_idx}/{len(self.loader_train)}, Time {tm.toc()}/{tm.hold()}, 训练完一个 batch: loss = {lss}, metric = {metric}\n")
                        if accumEpoch == int(len(self.args.CompressRateTrain)*len(self.args.SNRtrain)*self.args.epochs) and batch_idx==len(self.loader_train)-1:
                            with torch.no_grad():
                                for a, b, name in zip(hr, sr,filename):
                                    filename1 = '/home/jack/公共的/Python/PytorchTutor/lulaoshi/LeNet/image/origin/{}_hr.png'.format(name)
                                    data1 = a.permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                                    imageio.imwrite(filename1, data1)
                                    filename2 = '/home/jack/公共的/Python/PytorchTutor/lulaoshi/LeNet/image/net/{}_lr.png'.format(name)
                                    data2 = b.permute(1, 2, 0).type(torch.uint8).cpu().numpy()
                                    imageio.imwrite(filename2, data2)

                    # 学习率递减
                    self.optimizer.schedule()

                    # 计算并更新epoch的PSNR和MSE等metric
                    epochMetric = self.ckp.MeanMetricLog(compressrate, snr, len(self.loader_train))
                    # 计算并更新epoch的loss
                    epochLos = self.loss.mean_log(len(self.loader_train))

                    #print(f"\t训练完一个 Epoch: epochMetric = {epochMetric}, epochLos = {epochLos}, 用时:{tm.timer:.3f}/{tm.hold():.3f} \n")
                    self.ckp.write_log(f"\t训练完一个 Epoch: epochMetric = {epochMetric}, epochLos = {epochLos}, 用时:{tm.timer:.3f}/{tm.hold():.3f}", train=True)

                    # 断点可视化，在各个压缩率和信噪比下的Loss和PSNR，以及合并的loss
                    self.wr.WrTLoss(epochLos, int(self.ckp.LastSumEpoch+accumEpoch))
                    self.wr.WrTrainLoss(compressrate, snr, epochLos, epoch_idx)
                    
                    self.wr.WrLr(compressrate, snr, self.optimizer.get_lr(), epoch_idx)

                    self.wr.WrTrMetricOne(compressrate, snr, epochMetric, epoch_idx)
                    self.wr.WrTrainMetric(compressrate, snr, epochMetric, epoch_idx)
                    tm.reset()

                # 在每个 压缩率+信噪比 组合下都重置一次优化器
                self.optimizer.reset_state()

                # 在训练完每个压缩率和信噪比下的所有Epoch后,保存一次模型
                #self.ckp.saveModel(self, compressrate, snr, epoch=int(self.ckp.startEpoch+self.args.epochs))

        # 在训练完所有压缩率和信噪比后，保存损失日志
        self.ckp.saveLoss(self)
        # 在训练完所有压缩率和信噪比后，保存优化器
        self.ckp.saveOptim(self)
        # 在训练完所有压缩率和信噪比后，保存PSNR等指标日志
        self.ckp.save()
        self.ckp.write_log(f"#================================ 本次训练完毕,用时:{tm.hold()/60.0}分钟 =======================================",train=True)
        # 关闭日志
        self.ckp.done()
        print(f"====================== 关闭训练日志 {self.ckp.log_file.name} ===================================")
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print(color.fuchsia(f"\n#====================== 训练完毕,时刻:{now},用时:{tm.hold()/60.0}分钟 ==============================\n"))
        return

    def test(self):
        print(color.fuchsia(f"\n#================================ 开始测试 =======================================\n"))
        # 设置随机数种子
        #torch.manual_seed(self.args.seed)
        torch.set_grad_enabled(False)
        self.ckp.InittestDir(now=self.ckp.now)

        self.model.eval()
        self.model.model.eval()

        tm = utility.timer()
        #if self.args.save_results:
        #   self.ckp.begin_queue()

        print(f"共有{len(self.loader_test)}个数据集\n")
        self.ckp.write_log(f"共有{len(self.loader_test)}个数据集")

        # 依次遍历测试数据集
        for idx_data, ds in enumerate(self.loader_test):

            # 得到测试数据集名字
            DtSetName = ds.dataset.name
            print(f"数据集={DtSetName}, 长度={len(ds)}\n")
            self.ckp.write_log(f"开始在数据集{DtSetName}上测试")

            # 依次遍历压缩率
            for comprate_idx, compressrate in enumerate(self.args.CompressRateTrain):  #[0.17, 0.33]

                print(f"\t开始在 数据集为:{DtSetName}, 压缩率为:{compressrate} 下测试\n")
                # 写日志
                self.ckp.write_log(f"\t开始在 数据集为:{DtSetName}, 压缩率为:{compressrate} 下测试")

                # 初始化测试指标日志
                self.ckp.InitTestMetric(compressrate, DtSetName)

                # 依次遍历信噪比
                for snr_idx, snr in enumerate(self.args.SNRtest):   # [-6, -4, -2, 0, 2, 6, 10, 14, 18]

                    print(f"\t\t数据集为:{DtSetName}, 压缩率为:{compressrate} 信噪比为:{snr}\n")
                    # 写日志
                    self.ckp.write_log(f"\t\t数据集为:{DtSetName}, 压缩率为:{compressrate} 信噪比为:{snr}")

                    # 测试指标日志申请空间
                    self.ckp.AddTestMetric(compressrate, snr, DtSetName)

                    for batch_idx, (lr, hr, filename) in  enumerate(ds):

                        sr = self.model(hr, idx_scale=0, snr=snr, compr_idx=comprate_idx)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        # 保存图片
                        self.ckp.SaveTestFig(DtSetName, compressrate, snr, filename[0], sr)

                        # 计算bach内(测试时一个batch只有一张图片)的psnr和MSE
                        with torch.no_grad():
                            metric = utility.calc_metric(sr=sr, hr=hr, scale=1, rgb_range=self.args.rgb_range, metrics=self.args.metrics)
                            print(f"")
                        # 更新具体SNR下一张图片的PSNR和MSE等
                        self.ckp.UpdateTestMetric(compressrate, DtSetName,metric)
                        #print(f"数据集为:{DtSetName}, 压缩率为:{compressrate} 信噪比为:{snr},图片:{filename},指标:{}")

                        print(f"\t\t\t数据集:{DtSetName}({idx_data+1}/{len(self.loader_test)}),图片:{filename}({batch_idx+1}/{len(ds)}),压缩率:{compressrate}({comprate_idx+1}/{len(self.args.CompressRateTrain)}),信噪比:{snr}({snr_idx+1}/{len(self.args.SNRtest)}), 指标:{metric},时间:{tm.toc()}/{tm.hold()}")
                        self.ckp.write_log(f"\t\t\t数据集:{DtSetName}({idx_data+1}/{len(self.loader_test)}),图片:{filename}({batch_idx+1}/{len(ds)}),压缩率:{compressrate}({comprate_idx+1}/{len(self.args.CompressRateTrain)}),信噪比:{snr}({snr_idx+1}/{len(self.args.SNRtest)}), 指标:{metric},时间:{tm.toc()}/{tm.hold()}")

                    # 计算某个数据集下的平均指标
                    metrics = self.ckp.MeanTestMetric(compressrate, DtSetName,  len(ds))
                    self.wr.WrTestMetric(DtSetName, compressrate, snr, metrics)
                    self.wr.WrTestOne(DtSetName, compressrate, snr, metrics)
        self.ckp.write_log(f"===================================  测试结束 =======================================================")
        print(f"====================== 关闭测试日志  {self.ckp.log_file.name} ===================================")
        self.ckp.done()
        print(color.fuchsia(f"\n#================================ 完成测试, 用时:{tm.hold()/60.0}分钟 =======================================\n"))
        return


    def test1(self):  # 测试
        #  只要设置了torch.set_grad_enabled(False)那么接下来所有的tensor运算产生的新的节点都是不可求导的，
        #  这个相当于一个全局的环境，即使是多个循环或者是在函数内设置的调用，只要torch.set_grad_enabled(False)出现，
        # 则不管是在下一个循环里还是在主函数中，都不再求导，除非单独设置一个孤立节点，并把他的requires_grad设置成true。
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        #self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        print(f"trainer.pt  37: {len(self.loader_test)}, {len(self.scale)} \n")
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_queue()

        # ind1_scale = self.args.scale.index(1)  #  args.scale中1的索引

        for idx_data, d in enumerate(self.loader_test):
            i = 0
            print(color.higgreenfg_whitebg(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n  len(d) = {len(d)}\n"))  #   len(d) = 68
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # 对于测试数据集为Rain100L，去雨任务，忽略其他的scale，只针对sclae=1测试。
                if self.args.derain and d.dataset.name == 'Rain100L' and scale ==1:
                    #print(f"正在测试数据集:{d.dataset.name}, idx_scale = {idx_scale}, scale = {scale} \n")
                    for norain, rain, filename in tqdm(d, ncols=80):
                        #print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename}, norain.shape = {norain.shape}, rain.shape = {rain.shape} \n "))
                        norain,rain = self.prepare(norain, rain)
                        sr = self.model(rain, idx_scale, 15, 0.3)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, norain, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results_byQueue(d, filename[0], save_list, 1)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    isderain = 0
                # 对于测试数据集为 CBSD68，去噪任务，忽略其他的scale，只针对sclae=1测试。
                elif self.args.denoise and d.dataset.name == 'CBSD68' and scale == 1 :
                    print(f"正在测试数据集:{d.dataset.name}, idx_scale = {idx_scale}, scale = {scale} \n")
                    for hr, lr,filename in tqdm(d, ncols=80):
                        #print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename}, hr.shape = {hr.shape}, lr.shape = {lr.shape} \n "))
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level)
                        nois_hr = (noise+hr).clamp(0,255)
                        sr = self.model(nois_hr, idx_scale, 15, 0.3)
                        #print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename},sr.shape = {sr.shape}, hr.shape = {hr.shape}, lr.shape = {lr.shape} \n "))
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr( sr, hr, scale, self.args.rgb_range )
                        if self.args.save_results:
                            self.ckp.save_results_byQueue(d, filename[0], save_list, 50)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                elif d.dataset.name in ['Set1','Set2','Set3','Set5', 'Set14', 'B100', 'Urban100','DIV2K']:
                    print(f"正在测试数据集:{d.dataset.name}, idx_scale = {idx_scale}, scale = {scale}  \n")
                    for lr, hr, filename in tqdm(d, ncols=80):
                        #print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename}, lr.shape = {lr.shape}, hr.shape = {hr.shape} \n "))
                        # filename=('baby',),  lr.shape = torch.Size([1, 3, 256, 256]), hr.shape = torch.Size([1, 3, 512, 512])

                        lr, hr = self.prepare(lr, hr)
                        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename}, lr.shape = {lr.shape}, hr.shape = {hr.shape}"))
                        # lr.shape = torch.Size([1, 3, 256, 256]), hr.shape = torch.Size([1, 3, 512, 512])

                        sr = self.model(lr, idx_scale, 15, 1)

                        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename},   sr.shape = {sr.shape}"))
                        # sr.shape = torch.Size([1, 3, 512, 512])

                        sr = utility.quantize(sr, self.args.rgb_range)
                        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                        #   filename={filename},   sr.shape = {sr.shape}")) # sr.shape = torch.Size([1, 3, 512, 512])

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(sr, hr, scale, self.args.rgb_range)
                        #import pdb
                        #pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results_byQueue(d, filename[0], save_list, scale)
                        i = i+1
                    #print(f"line  = 137, idx_data = {idx_data}, idx_scale = {idx_scale}\n\
#self.ckp.log=\n{self.ckp.log}  \n")
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)

                    best = self.ckp.log.max(0)
                    #print(f"filename = {filename}, idx_data = {idx_data}, idx_scale = {idx_scale}\n\
#self.ckp.log=\n{self.ckp.log} \nbest = \n{best}  \n")
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        ))
                else:
                    print(f"d.dataset.name =  {d.dataset.name }, scale = {scale} \n")

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_queue()

        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

        torch.set_grad_enabled(True)









































