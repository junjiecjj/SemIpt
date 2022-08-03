
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""


import sys
import utility
import torch
from torch.autograd import Variable
from tqdm import tqdm


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

        self.wr.WrModel(self.model.model,torch.randn(16, 3, 48, 48))
        if self.args.load != '':
            if self.ckp.mark == True:
                self.optimizer.load(ckp.dir, epoch=ckp.startEpoch)

        self.error_last = 1e8


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
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
        print(color.fuchsia(f"\n#================================ 开始训练 =======================================\n"))
        torch.set_grad_enabled(True)
        ind1_scale = self.args.scale.index(1)

        tm = utility.timer()

        self.loader_train.dataset.set_scale(ind1_scale)
        #print(f"scale in train = {self.loader_train.dataset.scale[self.loader_train.dataset.idx_scale]}\n")
        AllEpoch = 0
        # 依次遍历压缩率
        for comprate_idx, compressrate in enumerate(self.args.CompressRateTrain):  #[0.17, 0.33, 0.4]
            # 依次遍历信噪比
            for snr_idx, snr in enumerate(self.args.SNRtrain): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
                print(color.fuchsia( f" 开始在压缩率索引为:{comprate_idx}, 压缩率为:{compressrate}， 信噪比索引为:{snr_idx}, 信噪比为:{snr} 下训练\n"))
                print(f"开始在压缩率索引为:{comprate_idx}, 压缩率为:{compressrate}， 信噪比索引为:{snr_idx}, 信噪比为:{snr} 下训练", file=self.ckp.log_file)
                # 初始化 特定信噪比和压缩率下 的Psnr日志
                self.ckp.InitPsnrLog(compressrate, snr)

                # 遍历epoch
                for epoch_idx in  range(self.ckp.startEpoch, self.ckp.startEpoch+self.args.epochs):
                    AllEpoch += 1
                    self.ckp.UpdateEpoch()

                    # 初始化loss日志
                    self.loss.start_log()

                    # 动态增加特定信噪比和压缩率下的Psnr日志
                    self.ckp.AddPsnrLog(compressrate, snr)

                    print(f"训练数据集的batch数 = {len(self.loader_train)}\n")

                    # 遍历训练数据集
                    for batch_idx, (lr, hr, filename)  in enumerate(self.loader_train):

                        #print(f"\nepoch_idx = {epoch_idx}, batch_idx = {batch_idx}, lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}\n")
                        #print(f"lr.shape = {lr.shape}, hr.shape = {hr.shape} \n")
                        print(f"\n Epoch {epoch_idx}/{self.ckp.startEpoch+self.args.epochs}, Iter {batch_idx}/{len(self.loader_train)}, Time {tm.toc()}/{tm.hold()} \n")
                        lr, hr = self.prepare(lr, hr)

                        self.optimizer.zero_grad()
                        sr = self.model(lr, idx_scale=ind1_scale, snr=snr, compr_idx=comprate_idx)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        # 计算batch内的loss
                        lss = self.loss(sr, hr)
                        #lss = Variable(lss, requires_grad = True)
                        lss.backward()
                        self.optimizer.step()

                        # 计算bach内的psnr
                        psnr = utility.calc_psnr(sr=sr, hr=hr, scale=1, rgb_range=self.args.rgb_range)

                        # 更新 bach内的psnr
                        self.ckp.UpdatePsnrLog(compressrate, snr, psnr)
                        print(f"\t\t训练完一个 batch: lr.shape = {lr.shape}, hr.shape = {hr.shape}, sr.shape = {sr.shape}, loss = {lss}, psnr = {psnr}\n")
                        print(f"\t\t训练完一个 batch: loss = {lss}, psnr = {psnr}\n", file=self.ckp.log_file)
                    # 计算并更新epoch的PSNR和LOSS
                    epochPsnr = self.ckp.meanPsnrLog(compressrate, snr, len(self.loader_train))
                    epochLos = self.loss.end_log(len(self.loader_train))
                    print(f"\t训练完一个 Epoch: epochPsnr = {epochPsnr}, epochLos = {epochLos} \n")
                    print(f"\t训练完一个 Epoch: epochPsnr = {epochPsnr}, epochLos = {epochLos}", file=self.ckp.log_file)
                    # 断点可视化，在各个压缩率和信噪比下的Loss和PSNR，以及合并的loss
                    self.wr.WrTLoss(epochLos, int(self.ckp.LastSumEpoch+AllEpoch))
                    self.wr.WrTrainLoss(compressrate, snr, epochLos, epoch_idx)

                    self.wr.WrTrPsnrOne(compressrate, snr, epochPsnr, epoch_idx)
                    self.wr.WrTrainPsnr(compressrate, snr, epochPsnr, epoch_idx)

                    # 学习率递减
                    self.optimizer.schedule()
                # 在每个 压缩率+信噪比 组合下都重置一次优化器
                self.optimizer.reset_state()

                # 在训练完每个压缩率和信噪比下的所有Epoch后,保存一次模型
                self.ckp.saveModel(self, compressrate, snr, epoch=int(self.ckp.startEpoch+self.args.epochs))

        # 在训练完所有压缩率和信噪比后，保存损失日志
        self.ckp.saveLoss(self)
        # 在训练完所有压缩率和信噪比后，保存优化器
        self.ckp.saveOptim(self)
        # 在训练完所有压缩率和信噪比后，保存PSNR等指标日志
        # 保存checkpoint日志
        self.ckp.save()
        print(f"#================================ 本次训练完毕 =======================================",file=self.ckp.log_file )
        # 关闭日志
        self.ckp.done()
        print(color.fuchsia(f"\n#================================ 训练完毕 =======================================\n"))

    def test(self):
        pass


    def test11(self):
        pass


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









































