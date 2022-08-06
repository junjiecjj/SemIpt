# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

# 本项目自己编写的库
# 参数
from option import args
# 数据集
import data
from data import  data_generator

# 损失函数
from loss.Loss import LOSS

# 训练器
from trainer import Trainer
#from  model import IPT

from model.__init__ import ModelSet

# Loss、PSNR等结果可视化等
from visual.summwriter import SummWriter


#  系统库
import torch
import utility
import warnings

warnings.filterwarnings('ignore')
import os
os.system('pip install einops')


# 设置随机数种子
torch.manual_seed( args.seed)


# 加载所有参数的类
ckp = utility.checkpoint(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#global model
if ckp.ok:
    # 数据迭代器，DataLoader
    loader = data_generator.DataGenerator(args)

    # 初始化模型
    if args.modelUse == 'ipt':
        _model = ModelSet[args.modelUse](args,ckp)
    elif args.modelUse == 'DeepSC':
        _model = ModelSet[args.modelUse](args)

    # 加载最初的预训练模型
    if args.pretrain != "":# 用预训练模型
        print(f"加载最原始的预训练模型\n")
        state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
        _model.model.load_state_dict(state_dict, strict=False)

    # 加载最近保存一次的的训练模型
    _model.load(ckp.get_path('model'), cpu=args.cpu)

    # 损失函数类
    los = LOSS(args, ckp)

    # tensorboard可视化
    wr = SummWriter(args)

    # 训练器，包括训练测试模块
    tr = Trainer(args, loader, _model, los, ckp, wr)

    # 训练
    if  args.wanttrain:
        print(f"I want to train \n")
        #tr.train()

    # 测试
    if  args.wanttest:
        tr.test()
        #print(f"I want to test \n")
        pass

    wr.WrClose()

    ckp.done()















































# def main():
#     #global model
#     if checkpoint.ok:
#         loader = data_generator.DataGenerator(args)
#         if args.modelUse == 'ipt':
#             _model = ModelSet[args.modelUse](args,checkpoint)
#         elif args.modelUse == 'DeepSC':
#             _model = ModelSet[args.modelUse](args )

#         if args.pretrain != "":# 用预训练模型
#             state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
#             _model.model.load_state_dict(state_dict, strict=False)

#         # args.test_only = false
#         _loss = loss.Loss(args, checkpoint) if not args.wanttest else None
#         t = Trainer(args, loader, _model, _loss, checkpoint)
#         if  args.wanttrain:
#             print(f"I want train \n")
#             t.train()
#         if  args.wanttest:
#             t.test1()
#             print(f"I want test \n")
#         checkpoint.done()


# if __name__ == '__main__':
#     main()





























