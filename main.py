
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

# 本项目自己编写的库
from option import args
import data
from loss.Loss import LOSS
from trainer import Trainer
#from  model import IPT

from model.__init__ import ModelSet
from data import    data_generator


#  系统库
import torch
import utility
import warnings

warnings.filterwarnings('ignore')
import os
os.system('pip install einops')
torch.manual_seed(args.seed)


checkpoint = utility.checkpoint(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#global model
if checkpoint.ok:
    # 数据迭代器，DataLoader
    loader = data_generator.DataGenerator(args)

    # 初始化模型
    if args.modelUse == 'ipt':
        _model = ModelSet[args.modelUse](args,checkpoint)
    elif args.modelUse == 'DeepSC':
        _model = ModelSet[args.modelUse](args)

    # 加载最初的预训练模型
    if args.pretrain != "":# 用预训练模型
        print(f"用最原始的预训练模型\n")
        state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
        _model.model.load_state_dict(state_dict, strict=False)

    # 损失函数类
    los = LOSS(args, checkpoint)

    # 训练器，包括训练测试模块
    tr = Trainer(args, loader, _model, los, checkpoint)

    # 训练
    if  args.wanttrain:
        print(f"I want train \n")
        tr.train()

    # 测试
    if  args.wanttest:
        #tr.test1()
        #print(f"I want test \n")
        pass
    checkpoint.done()















































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





























