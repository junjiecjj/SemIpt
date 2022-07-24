# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import sys
import utility
import torch
from torch.autograd import Variable
from tqdm import tqdm


# 本项目自己编写的库
from trainer import Trainer
from ColorPrint  import ColoPrint
color = ColoPrint()
print(color.fuchsia("Color Print Test Pass"))

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        print(f"trainer  self.scale = {self.scale} \n")
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.error_last = 1e8


    def loadmodel(self):
        # 初始化模型
        if args.modelUse == 'ipt':
            _model = ModelSet[args.modelUse](args,checkpoint)
        elif args.modelUse == 'DeepSC':
            _model = ModelSet[args.modelUse](args， compressRate)

        # 加载最初的预训练模型
        if args.pretrain != "":# 用预训练模型
            print(f"用最原始的预训练模型\n")
            state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
            _model.model.load_state_dict(state_dict, strict=False)



    def test(self):

        #  只要设置了torch.set_grad_enabled(False)那么接下来所有的tensor运算产生的新的节点都是不可求导的，
        #  这个相当于一个全局的环境，即使是多个循环或者是在函数内设置的调用，只要torch.set_grad_enabled(False)出现，
        # 则不管是在下一个循环里还是在主函数中，都不再求导，除非单独设置一个孤立节点，并把他的requires_grad设置成true。
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        print(f"trainer.pt  37: {len(self.loader_test)}, {len(self.scale)} \n")
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results:
            self.ckp.begin_background()

        # print(color.higgreenfg_whitebg(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #       len(self.loader_test) = {len(self.loader_test)}\n"))  #   len(self.loader_test) = 5
        for idx_data, d in enumerate(self.loader_test):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                #print(color.higgreenfg_whitebg(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                #                               idx_scale = {idx_scale}, scale = {scale}"))
                # idx_scale = 0, scale = 2
                print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                d = {d}\n"))

                d.dataset.set_scale(idx_scale)

                if self.args.derain: # 不进入此处
                    for norain, rain, filename in tqdm(d, ncols=80):
                        # print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
# filename={filename}, norain.shape = {norain.shape}, rain.shape = {rain.shape} \n "))
# filename=('rain-051',), norain.shape = torch.Size([1, 3, 321, 481]), rain.shape = torch.Size([1, 3, 321, 481])
                        norain,rain = self.prepare(norain, rain)
                        sr = self.model(rain, idx_scale,15)
                        #print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
# filename={filename}, sr.shape = {sr.shape} \n "))
# filename=('rain-051',), sr.shape = torch.Size([1, 3, 321, 481])
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, norain, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 1)
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
                elif self.args.denoise: # 不进入此处
                    for hr, _,filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise+hr).clamp(0,255)
                        sr = self.model(nois_hr, idx_scale,15)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 50)

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
                else:
                    for lr, hr, filename in tqdm(d, ncols=80):
                        print(color.higgreenfg_whitebg(f"\n File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n filename={filename}, lr.shape = {lr.shape}, hr.shape = {hr.shape} \n "))
                        # filename=('baby',),  lr.shape = torch.Size([1, 3, 256, 256]), hr.shape = torch.Size([1, 3, 512, 512])

                        lr, hr = self.prepare(lr, hr)
                        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                        # filename={filename}, lr.shape = {lr.shape}, hr.shape = {hr.shape}"))
                        # lr.shape = torch.Size([1, 3, 256, 256]), hr.shape = torch.Size([1, 3, 512, 512])

                        sr = self.model(lr, idx_scale,15)

                        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                        #   filename={filename},   sr.shape = {sr.shape}")) # sr.shape = torch.Size([1, 3, 512, 512])

                        sr = utility.quantize(sr, self.args.rgb_range)
                        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                        #   filename={filename},   sr.shape = {sr.shape}")) # sr.shape = torch.Size([1, 3, 512, 512])

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr( sr, hr, scale, self.args.rgb_range)
                        #import pdb
                        #pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
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

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
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
        torch.set_grad_enabled(True)
        ind1_scale = self.args.scale.index(1)
        self.loader_train.dataset.set_scale(ind1_scale)
        # 依次遍历压缩率
        for comprate_idx, compressrate in enumerate(self.args.CompressRateTrain):  #[0.17, 0.33, 0.4]

            self.model = self.loadmodel()
            self.loss = my_loss
            self.optimizer = utility.make_optimizer(args, self.model)
            if self.args.load != '':
                self.optimizer.load(ckp.dir, epoch=len(ckp.psnrlog))

            # 依次遍历信噪比
            for snr_idx, snr in enumerate(self.args.trainSNR): # [-6, -4, -2, 0, 2, 6, 10, 14, 18]
                print(f"\ncomprate_idx = {comprate_idx}, compressrate = {compressrate}， snr_idx = {snr_idx}, snr = {snr}, \n")

                # 遍历epoch
                for epoch_idx in  range(self.args.epochs):

                    # 遍历训练数据集
                    for batch_idx, (lr, hr, filename)  in tqdm(enumerate(self.loader_train), ncols=80):
                        print(f"\nepoch_idx = {epoch_idx}, batch_idx = {batch_idx}, lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}\n")


                        # lr, hr = self.prepare(lr, hr)

                        # self.optimizer.zero_grad()
                        # sr = self.model(lr, idx_scale=0, snr=snr, compressrate=compressrate)
                        # sr = utility.quantize(sr, self.args.rgb_range)

                        # lss = self.loss(sr, hr)
                        # lss = Variable(lss, requires_grad = True)

                        # lss.backward()
                        # self.optimizer.step()

                        # if epoch_idx%10 == 0:
                        #     print(f"\n[SNR={snr} CompressaRate = {comprate}] Epoch:{epoch_idx}/{self.args.epochs} Iter:{batch_idx}/{len(self.loader_train)} ")
                        #     print(f"\n lr.shape = {lr.shape}, hr.shap = {hr.shape}, filename = {filename} \n")


    def test1(self):  # 测试
        #  只要设置了torch.set_grad_enabled(False)那么接下来所有的tensor运算产生的新的节点都是不可求导的，
        #  这个相当于一个全局的环境，即使是多个循环或者是在函数内设置的调用，只要torch.set_grad_enabled(False)出现，
        # 则不管是在下一个循环里还是在主函数中，都不再求导，除非单独设置一个孤立节点，并把他的requires_grad设置成true。
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
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

                        sr = self.model(lr, idx_scale, 15, 0.3)

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









































