
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""

# 本项目自己编写的库
from option import args
import data
import loss
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



def main():
    #global model
    if checkpoint.ok:
        loader = data_generator.DataGenerator(args)
        if args.modelUse == 'ipt':
            _model = ModelSet[args.modelUse](args,checkpoint)
        elif args.modelUse == 'DeepSC':
            _model = ModelSet[args.modelUse](args )

        if args.pretrain != "":# 用预训练模型
            state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
            _model.model.load_state_dict(state_dict, strict=False)

        # args.test_only = false
        _loss = loss.Loss(args, checkpoint) if not args.wanttest else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        if  args.wanttrain:
            print(f"I want train \n")
            t.train()
        if  args.wanttest:
            t.test1()
            print(f"I want test \n")
        checkpoint.done()


# if __name__ == '__main__':
#     main()


#global model
if checkpoint.ok:
    loader = data_generator.DataGenerator(args)
    if args.modelUse == 'ipt':
        _model = ModelSet[args.modelUse](args,checkpoint)
    elif args.modelUse == 'DeepSC':
        _model = ModelSet[args.modelUse](args )

    if args.pretrain != "":# 用预训练模型
        print(f"用最原始的预训练模型\n")
        state_dict = torch.load(args.pretrain, map_location=torch.device('cpu'))
        _model.model.load_state_dict(state_dict, strict=False)

    # args.test_only = false
    los = loss.Loss(args, checkpoint) if not args.wanttest else None
    tr = Trainer(args, loader, _model, los, checkpoint)
    if  args.wanttrain:
        print(f"I want train \n")
        #tr.train()
    if  args.wanttest:
        tr.test1()
        #print(f"I want test \n")
        pass
    checkpoint.done()






























