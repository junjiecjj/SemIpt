# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import sys,os
sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()

import glob
import random
import pickle
import io

import PIL.Image as pil_image
from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import torchvision.transforms as tfs

def search(root, target="JPEG"):
    item_list = []
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            #print('[-]', path)
            item_list.extend(search(path, target))
        elif path.split('/')[-1].startswith(target):
            item_list.append(path)
        elif path.split('/')[-2] == target or path.split('/')[-3] == target or path.split('/')[-4] == target:
            item_list.append(path)
        else:
            ttt = 1
            #print('[!]', path)
    return item_list

class SRData(data.Dataset):
    #  test:  args, name = 'Set5',    train=False, benchmark=True
    #  test:  args, name = 'DIV2K',   train=False, benchmark=False
    #  train: args, name = 'DIV2K',   train=True,  benchmark=False
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale  #  [2,3,4]
        self.idx_scale = 0

        if self.args.derain:# 不进入此处
            self.derain_test = os.path.join(args.dir_data, "Rain100L")
            self.derain_lr_test = search(self.derain_test, "rain")
            self.derain_hr_test = [path.replace("rainy/","no") for path in self.derain_lr_test]
            #print(color.higgreenfg_whitebg( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n\
    #self.derain_test = {self.derain_test}\n self.derain_lr_test = {self.derain_lr_test}\n self.derain_hr_test = {self.derain_hr_test}"))

        # self.derain_test = /home/jack/IPT-Pretrain/Data/Rain100L
        #  self.derain_lr_test = [
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/rainy/rain-051.png',
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/rainy/rain-068.png',
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/rainy/rain-025.png',
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/rainy/rain-052.png', png',....]  len(self.derain_lr_test)=100
        #  self.derain_hr_test = [
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/norain-051.png',
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/norain-068.png',
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/norain-025.png',
        #    '/home/jack/IPT-Pretrain/Data/Rain100L/norain-052.png', ..... ] len(self.derain_hr_test)=100


        self._set_filesystem(args.dir_data)
        print(color.higyellowfg_whitebg( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n self.apath = {self.apath}\n self.dir_hr = {self.dir_hr}\n self.dir_lr = {self.dir_lr}"))
        # self.apath = /home/jack/IPT-Pretrain/Data/benchmark/Set5
        # self.dir_hr = /home/jack/IPT-Pretrain/Data/benchmark/Set5/HR
        # self.dir_lr = /home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic

        # self.apath = /home/jack/IPT-Pretrain/Data/DIV2K
        # self.dir_hr = /home/jack/IPT-Pretrain/Data/DIV2K/DIV2K_train_HR
        # self.dir_lr = /home/jack/IPT-Pretrain/Data/DIV2K/DIV2K_train_LR_bicubic

        # self.apath = /home/jack/IPT-Pretrain/Data/benchmark/Rain100L
        # self.dir_hr = /home/jack/IPT-Pretrain/Data/benchmark/Rain100L/HR
        # self.dir_lr = /home/jack/IPT-Pretrain/Data/benchmark/Rain100L/LR_bicubic

        # self.apath = /home/jack/IPT-Pretrain/Data/benchmark/CBSD68
        # self.dir_hr = /home/jack/IPT-Pretrain/Data/benchmark/CBSD68/HR
        # self.dir_lr = /home/jack/IPT-Pretrain/Data/benchmark/CBSD68/LR_bicubic

        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')   # path_bin = /home/jack/IPT-Pretrain/Data/benchmark/Set5/bin
            os.makedirs(path_bin, exist_ok=True)
            #print(color.higyellowfg_whitebg( f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n \
            # path_bin = {path_bin}\n "))
            # path_bin = /home/jack/IPT-Pretrain/Data/benchmark/Set5/bin
            # path_bin = /home/jack/IPT-Pretrain/Data/benchmark/Set14/bin
            # path_bin = /home/jack/IPT-Pretrain/Data/benchmark/B100/bin
            # path_bin = /home/jack/IPT-Pretrain/Data/benchmark/Urban100/bin

            # path_bin = /home/jack/IPT-Pretrain/Data/DIV2K/bin

        list_hr, list_lr = self._scan()
        print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        \n name = {self.name} len(list_hr) = {len(list_hr)}, len(list_lr) = {len(list_lr)}, len(list_lr[0]) = {len(list_lr[0])} \n "))
        #   name = Set5     len(list_hr) = 5, len(list_lr) = 3, len(list_lr[0]) = 5  test
        #   name = Set14    len(list_hr) = 14, len(list_lr) = 3, len(list_lr[0]) = 14 test
        #   name = B100     len(list_hr) = 100, len(list_lr) = 3, len(list_lr[0]) = 100 test
        #   name = Urban100 len(list_hr) = 100, len(list_lr) = 3, len(list_lr[0]) = 100  test
        #   name = DIV2K    len(list_hr) = 10, len(list_lr) = 3, len(list_lr[0]) = 10  test

        #  name = Rain100L len(list_hr) = 0, len(list_lr) = 3, len(list_lr[0]) = 0   train
        #  name = CBSD68   len(list_hr) = 0, len(list_lr) = 1, len(list_lr[0]) = 0


        # name = DIV2K len(list_hr) = 800, len(list_lr) = 3, len(list_lr[0]) = 800   train


        # if self.name == 'Set5':
        #      print(f"list_hr = {list_hr},\n\nlist_lr = {list_lr}")

        if args.ext.find('img') >= 0 or benchmark:
            print("1  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            print("2  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            # print(color.higfuchsiafg_whitebg( f"self.dir_hr.replace(self.apath, path_bin) = {self.dir_hr.replace(self.apath, path_bin)}"))
            # self.dir_hr.replace(self.apath, path_bin) = /home/jack/IPT-Pretrain/Data/DIV2K/bin/DIV2K_train_HR
            os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
            for s in self.scale:
                if s == 1:
                    os.makedirs( os.path.join(self.dir_hr), exist_ok=True)
                else:
                    # print(f"{os.path.join(self.dir_lr.replace(self.apath, path_bin),'X{}'.format(s))}")
                    # /home/jack/IPT-Pretrain/Data/DIV2K/bin/DIV2K_train_LR_bicubic/X2
                    # /home/jack/IPT-Pretrain/Data/DIV2K/bin/DIV2K_train_LR_bicubic/X3
                    # /home/jack/IPT-Pretrain/Data/DIV2K/bin/DIV2K_train_LR_bicubic/X4
                    os.makedirs(os.path.join(self.dir_lr.replace(self.apath, path_bin),'X{}'.format(s)), exist_ok=True)

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                # h = /cache/data/DIV2K/HR/baboon.png
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                # b = /cache/data/DIV2K/bin/HR/baboon.pt
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True)


            for i, ll in enumerate(list_lr):
                for l in ll:
                    # l = /cache/data/DIV2K/LR_bicubic/baboonx2.png
                    # b = /cache/data/DIV2K/bin/LR_bicubic/baboonx2.png
                    b = l.replace(self.apath, path_bin)
                    # b = /cache/data/DIV2K/bin/LR_bicubic/baboonx2.pt
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
# \n name = {self.name} len(self.images_hr) = {len(self.images_hr)}, len(self.images_lr) = {len(self.images_lr)}, len(self.images_lr[0]) = {len(self.images_lr[0])} \n "))
        if train:
            print(f"train ={train} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            n_patches = args.batch_size * args.test_every # 16*1000
            n_images = len(args.data_train) * len(self.images_hr)

            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)
            # print(f"n_patches = {n_patches}, n_images = {n_images} repead = {self.repeat}\n")
            # n_patches = 16000, n_images = 800 repead = 20
        if self.args.derain:
            self.images_hr, self.images_lr = self.derain_hr_test, self.derain_lr_test
        if self.args.denoise:
            self.images_hr = glob.glob(os.path.join(self.apath, '*.png'))
            self.images_lr = glob.glob(os.path.join(self.apath, '*.png'))
            print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n\
                name = {self.name} len(self.images_hr) = {len(self.images_hr)}, len(self.images_lr) = {len(self.images_lr)}  \n "))


    # Below functions as used to prepare images
    def _scan(self):
        #  列表，列表的每个元素为“每张图片的路径+文件名”  /cache/data/DIV2K/HR/baboon.png
        names_hr = sorted( glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0])) )
        # name_hr =
        # ['/home/jack/IPT-Pretrain/Data/benchmark/Set5/HR/baby.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/HR/bird.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/HR/butterfly.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/HR/head.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/HR/woman.png'],
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                if s != 1:
                    names_lr[si].append(os.path.join(self.dir_lr, 'X{}/{}x{}{}'.format(s, filename, s, self.ext[1] )))
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si]=names_hr
        print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        \n names_hr = {names_hr},names_lr = {names_lr} \n"))

        # names_lr是列表的列表，每个子列表的每个元素为“'/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X2/babyx2.png'”
        # names_lr =  [
        #  ['/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X2/babyx2.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X2/birdx2.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X2/butterflyx2.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X2/headx2.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X2/womanx2.png'],
        #  ['/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X3/babyx3.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X3/birdx3.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X3/butterflyx3.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X3/headx3.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X3/womanx3.png'],
        #  ['/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X4/babyx4.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X4/birdx4.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X4/butterflyx4.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X4/headx4.png', '/home/jack/IPT-Pretrain/Data/benchmark/Set5/LR_bicubic/X4/womanx4.png']
        # ]

        # Rain100L   names_hr = [],names_lr = [[]]
        #  CBSD68    names_hr = [],names_lr = [[]]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        # dir_data = '/home/jack/IPT-Pretrain/Data/'
        self.apath = os.path.join(dir_data, self.name)       # /home/jack/IPT-Pretrain/Data/DIV2K
        self.dir_hr = os.path.join(self.apath, 'HR')         # /home/jack/IPT-Pretrain/Data/DIV2K/HR
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')  #  /home/jack/IPT-Pretrain/Data/DIV2K/LR_bicubic
        # if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')


    """
    打开文件并读取 /cache/data/DIV2K/LR_bicubic/baboonx2.png
    并保存为二进制：/cache/data/DIV2K/bin/LR_bicubic/baboonx2.pt
    """
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            print("iiiiiiiiiiiiiiiiiiiiiiiiiii\n")
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        if self.args.derain:  # 不进入此处
            norain, rain, filename = self._load_rain_test(idx)
            #print(color.higyellowfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\nnorain.shape = {norain.shape}, rain.shape = {rain.shape}, filename = {filename}\n"))
    #  norain.shape = (321, 481, 3), rain.shape = (321, 481, 3), filename = rain-051
            pair = common.set_channel(*[norain, rain], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

            return pair_t[0], pair_t[1], filename
        if self.args.denoise: # 不进入此处
            hr, filename = self._load_file_hr(idx)
            pair = self.get_patch_hr(hr)
            pair = common.set_channel(*[pair], n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
            return pair_t[0],pair_t[0], filename

        # 默认，图像缩放任务
        lr, hr, filename = self._load_file(idx)
        #print(color.higredfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #   lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}\n"))
        #  lr.shape = (256, 256, 3), hr.shape = (512, 512, 3), filename = baby

        pair = self.get_patch(lr, hr)
        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #   pair[0].shape = {pair[0].shape}, pair[1].shape = {pair[1].shape}\n"))
        #  pair[0].shape = (256, 256, 3), pair[1].shape = (512, 512, 3)

        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        #print(color.higyellowfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #   pair[0].shape = {pair[0].shape}, pair[1].shape = {pair[1].shape}\n"))
        #  pair[0].shape = (256, 256, 3), pair[1].shape = (512, 512, 3)

        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)  # rgb_range=255
        #print(color.higcyanfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    pair_t[0].shape = {pair_t[0].shape}, pair_t[1].shape = {pair_t[1].shape}\n"))
        #  pair_t[0].shape = torch.Size([3, 256, 256]), pair_t[1].shape = torch.Size([3, 512, 512])

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            print(color.higyellowfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            len(self.images_hr) * self.repeat = {len(self.images_hr) * self.repeat} \n"))
            # len(self.images_hr) * self.repeat = 16000
            return len(self.images_hr) * self.repeat
        else:
            if self.args.derain:
                return int(len(self.derain_hr_test)/self.args.derain_test)
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            #if self.args.deblur:
            #    return random.randint(0, len(self.deblur_hr_test) - 1)
            #if self.args.dehaze:
            #    return random.randint(0, len(self.haze_test) - 1)
            return idx

    def _load_file_hr(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)

        return hr, filename

    def _load_rain_test(self, idx):
        f_hr = self.derain_hr_test[idx]
        f_lr = self.derain_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename

    def _load_file(self, idx):
        idx = self._get_index(idx)
        #print(color.higcyanfg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    \n idx = {idx}\n" ))
        #print(color.higfuchsiafg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    \n idx = {idx}\n" ))
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        #print(color.higfuchsiafg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    \n f_hr = {f_hr}, f_lr = {f_lr}, filename = {filename}\n" ))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
            #print(color.higgreenfg_whitebg(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #        lr.shape = {lr.shape}, hr.shape = {hr.shape}, filename = {filename}"))
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename



    def get_patch_hr(self, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            hr = self.get_patch_img_hr(
                hr,
                patch_size=self.args.patch_size,
                scale=1
            )

        return hr

    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        ih, iw = img.shape[:2]

        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy:iy + ip, ix:ix + ip, :]

        return ret

    def get_patch_cjj_for_test(self, lr, hr):
        scale = self.scale[self.idx_scale]
        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #  lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))
        #  lr.shape = (256, 256, 3), hr.shape = (512, 512, 3), self.scale = [2, 3, 4],scale = 2
        if  True:
            print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            name = {self.name} lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))
            lr, hr = common.get_patch(lr, hr,
                patch_size=self.args.patch_size*scale,   # 48*scale
                scale=scale,
                multi=(len(self.scale) > 1)
            )
            print(color.higcyanfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            name = {self.name}  lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))

            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
                print(color.higbluefg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            name = {self.name}   lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))
        else:
            print("\nhhhhhhhhhhhhhhhh\n")
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    lr.shape = {lr.shape}, hr.shape = {hr.shape},\n"))
        return lr, hr

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))
        #  lr.shape = (256, 256, 3), hr.shape = (512, 512, 3), self.scale = [2, 3, 4],scale = 2
        if  self.train:
            #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #    lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size*scale,
                scale=scale,
                multi=(len(self.scale) > 1)
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
            #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #    lr.shape = {lr.shape}, hr.shape = {hr.shape}, self.scale = {self.scale},scale = {scale}\n"))
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    lr.shape = {lr.shape}, hr.shape = {hr.shape},\n"))
        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
                idx_scale = {idx_scale} \n"))
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

