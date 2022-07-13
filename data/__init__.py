# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""




from importlib import import_module
#from dataloader import MSDataLoader
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset
import sys,os
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()


# This is a simple wrapper function for ConcatDataset
class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)
        self.train = datasets[0].train

    def set_scale(self, idx_scale):
        print(color.higyellowfg_whitebg( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n\
    idx_scale = {idx_scale} \n"))
        for d in self.datasets:
            if hasattr(d, 'set_scale'): d.set_scale(idx_scale)

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            datasets = []
            # for d in args.data_train:
            #     module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
            #     m = import_module('data.' + module_name.lower())
            #     datasets.append(getattr(m, module_name)(args, name=d))
            module_name = args.data_train[0] if args.data_train[0].find('DIV2K-Q') < 0 else 'DIV2KJPEG'
            m = import_module('data.' + module_name.lower())
            datasets = getattr(m, module_name)(args, name=module_name)
            self.loader_train = dataloader.DataLoader(
                # MyConcatDataset(datasets),
                # datasets[0],
                datasets,
                batch_size=args.batch_size,  # 16
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        self.loader_test = []
        if not args.train_only:
            for d in args.data_test:
            # print(color.higyellowfg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #                    \n d = {d}" ))
                if d in ['Set1','Set2','Set3','Set5', 'Set14', 'B100', 'Urban100', 'CBSD68','Rain100L']:
                    m = import_module('data.benchmark')
                    testset = getattr(m, 'Benchmark')(args, train=False, name=d)
                else:
                    module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                    m = import_module('data.' + module_name.lower())
                    # testset 为class
                    testset = getattr(m, module_name)(args, train=False, name=d)

                self.loader_test.append(
                    dataloader.DataLoader(
                        testset,
                        batch_size=  args.test_batch_size,  #  1
                        shuffle=False,
                        pin_memory=not args.cpu,
                        num_workers=args.n_threads,
                    )
                )
