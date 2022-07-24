
# -*- coding: utf-8 -*-
"""
Created on 2022/07/07

@author: Junjie Chen

"""


import sys,os
#sys.path.append(os.getcwd())
from model  import common
# 或
# from .  import common

sys.path.append("..")
from  ColorPrint import ColoPrint
color =  ColoPrint()



import math
import torch
import torch.nn.functional as F
import torch.nn.parallel as P
from torch import nn, Tensor
from einops import rearrange
import copy




def make_model(args, parent=False):
    return ipt(args)

class ipt(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ipt, self).__init__()
        print(f"initialing IPT Model.....\n")
        # print(f"current =  {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
        self.scale_idx = 0
        self.snr = 10
        self.compRate = 0.12

        self.args = args
        # print(f"In Ipt, args.scale = {args.scale} \n")
        n_feats = args.n_feats  # number of feature maps = 64
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(args.rgb_range)  # rgb_range = 255
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.ModuleList([
            nn.Sequential(
                conv(args.n_colors, n_feats, kernel_size),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ])

        """
        img_dim=48, patch_dim=3, num_channels=64, embedding_dim=64*3*3 ,num_heads=12,num_layers=12,
        hidden_dim=64*3*3*4,num_queries=1,dropout_rate=0,mlp=false,pos_every=false,no_pos=false,
        no_norm=false
        """
        self.body = VisionTransformer(img_dim=args.patch_size, patch_dim=args.patch_dim,
                                    num_channels=n_feats,
                                    embedding_dim=n_feats*args.patch_dim*args.patch_dim,
                                    num_heads=args.num_heads, num_layers=args.num_layers,
                                    hidden_dim=n_feats*args.patch_dim*args.patch_dim*4,
                                    num_queries = args.num_queries, dropout_rate=args.dropout_rate,
                                    mlp=args.no_mlp ,pos_every=args.pos_every,no_pos=args.no_pos,
                                    no_norm=args.no_norm)

        self.tail = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, s, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ) for s in args.scale
        ])


    def forward(self, x):
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #     \n x.shape = {x.shape}"))  # x.shape = torch.Size([1, 3, 48, 48])
        x = self.sub_mean(x)
        #print(color.fuchsia(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n sub mean x.shape = {x.shape}"))  # x.shape = torch.Size([1, 3, 48, 48])
        x = self.head[self.scale_idx](x)
        #print(color.fuchsia(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #       \n after head x.shape = {x.shape}"))  # x.shape = torch.Size([1, 64, 48, 48])

        # print(f"In IPT ipt snr = {self.snr}\n")

        res = self.body(x, self.scale_idx, self.snr, self.compRate)
        #print(color.fuchsia(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #    \n after body x.shape = {x.shape}, self.scale_idx={self.scale_idx}"))  # x.shape = torch.Size([1, 64, 48, 48])  self.scale_idx=0
        res += x

        x = self.tail[self.scale_idx](res)
        #print(color.fuchsia(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n after tail x.shape = {x.shape}")) # x.shape = torch.Size([1, 3, 96, 96])
        x = self.add_mean(x)
        #print(color.fuchsia(f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n add mean x.shape = {x.shape}")) # x.shape = torch.Size([1, 3, 96, 96])
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        #print(color.higgreenfg_whitebg(f"\nFile={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #        scale_idx = {scale_idx} \n"))
    def set_snr(self, snr):
        self.snr = snr

    def set_comprate(self, compressrate):
        self.compRate = compressrate

"""
img_dim=48, patch_dim=3, num_channels=64, embedding_dim=64*3*3 ,num_heads=12,num_layers=12,
hidden_dim=64*3*3*4,num_queries=1,dropout_rate=0,mlp=false,pos_every=false,no_pos=false,
no_norm=false
"""
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0  #  64*3*3/12
        assert img_dim % patch_dim == 0     #  48/3
        self.no_norm = no_norm              #  false
        self.mlp = mlp                      #  false
        self.embedding_dim = embedding_dim  #  64*3*3 = 576
        self.num_heads = num_heads          #  12
        self.patch_dim = patch_dim          #  3
        self.num_channels = num_channels    #  64

        self.img_dim = img_dim              #  48
        self.pos_every = pos_every          #  false
        self.num_patches = int((img_dim // patch_dim) ** 2)      #  256
        self.seq_length = self.num_patches                       #  256
        self.flatten_dim = patch_dim * patch_dim * num_channels  #   576

        self.out_dim = patch_dim * patch_dim * num_channels      #   576

        self.no_pos = no_pos       # false

        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)  #  576,  576
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),  # 576,  64*3*3*4=2304
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),  #  2304, 576
                nn.Dropout(dropout_rate)
            )

            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)  # 6, 576*256=147456

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        # embedding_dim=576, num_heads=12, hidden_dim=2304, dropout_rate=0, self.no_norm=false
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)


        if not self.no_pos:
            #  256, 576, 256
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #      \n self.seq_length = {self.seq_length}, self.embedding_dim={self.embedding_dim},self.seq_length={self.seq_length}"))
            # self.seq_length = 256, self.embedding_dim=576,self.seq_length=256
        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, query_idx, snr, CompRate, con=False):
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  # x.shape = torch.Size([1, 64, 48, 48])
        x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  # x.shape = torch.Size([256, 1, 576])

        if self.mlp==False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            query_embed = self.query_embed.weight[query_idx].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #    \n x.shape = {x.shape}, query_embed.shape = {query_embed.shape}, {self.query_embed.weight.shape}"))
            # x.shape = torch.Size([256, 1, 576]), query_embed.shape = torch.Size([256, 1, 576])
        else:
            query_embed = None

        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n x.shape = {x.shape}, query_embed.shape = {query_embed.shape}"))
        # x.shape = torch.Size([256, 1, 576])  query_embed.shape = torch.Size([256, 1, 576])

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)
            # pos.shape = # (1,576,256)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n pos.shape = {pos.shape}"))  # pos.shape = torch.Size([256, 1, 576])

        if self.pos_every:
            x = self.encoder(x, pos=pos)
            x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            x = self.decoder(x, x, query_pos=query_embed)
        else:
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n (x+pos).shape = {(x+pos).shape}"))  # (x+pos).shape = torch.Size([256, 1, 576])
            x = self.encoder(x+pos)
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  # x.shape = torch.Size([256, 1, 576])
            #print(f"In IPT VisionTransformer forward snr = {snr} ")

            # print(f"\n In IPT VisionTransformer forward Compress rare = {CompRate}\n")
            x = common.awgn(x, snr)

            x = self.decoder(x, x, query_pos=query_embed)
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  # x.shape = torch.Size([256, 1, 576])

        if self.mlp==False:
            x = self.mlp_head(x) + x
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  # pos.shape = torch.Size([256, 1, 576])
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  #  x.shape = torch.Size([1, 256, 576])
        if con:
            print("i'am in con ........")
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}"))  # x.shape = torch.Size([1, 64, 48, 48])
        return x

class LearnedPositionalEncoding(nn.Module):
    #  256, 576, 256
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim) # 256, 576
        self.seq_length = seq_length

        self.register_buffer("position_ids", torch.arange(self.seq_length).expand((1, -1)) )
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n position_ids.shape = {self.position_ids.shape}"))  # position_ids.shape = torch.Size([1, 256])

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
            #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
            #  \n position_ids.shape = { position_ids.shape}"))  # position_ids.shape = torch.Size([1, 256])

        position_embeddings = self.pe(position_ids) # (1,256,576)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n position_embeddings.shape = {position_embeddings.shape}"))  # position_embeddings.shape = torch.Size([1, 256, 576])
        return position_embeddings

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)
        return output

"""
d_model=64*3*3=576 ,nhead=12, dim_feedforward=64*3*3*4,dropout =0,no_norm=false
"""
# # 此模块不改变x的shape，即输入和输出的shape一样
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                activation="relu"):
        super().__init__()
        #  值得注意的一点是，使用系统库nn.MultiheadAttention而不是自己实现的模块时,query，key，value的
        # 输入形状一定是 [sequence_size, batch_size, emb_size],
        # 除非使用参数batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)  #  576, 2304
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos = None):
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src.shape = {src.shape}, pos = {pos}"))  # src.shape = torch.Size([256, 1, 576]), pos = None
        src2 = self.norm1(src)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src2.shape = {src2.shape}"))  # src2.shape = torch.Size([256, 1, 576])
        q = k = self.with_pos_embed(src2, pos)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n q.shape = {q.shape}"))  # q.shape = torch.Size([256, 1, 576])
        src2 = self.self_attn(q, k, src2)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src2[0].shape = {src2[0].shape}, src2[1].shape = {src2[1].shape}"))
        # src2[0].shape = torch.Size([256, 1, 576]), src2[1].shape = torch.Size([1, 256, 256])
        src = src + self.dropout1(src2[0])
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src.shape = {src.shape}"))  # src.shape = torch.Size([256, 1, 576])
        src2 = self.norm2(src)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src2.shape = {src2.shape}"))  # src2.shape = torch.Size([256, 1, 576])
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src2.shape = {src2.shape}"))  # src2.shape = torch.Size([256, 1, 576])
        src = src + self.dropout2(src2)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n src.shape = {src.shape}"))  # src.shape = torch.Size([256, 1, 576])
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output


class TransformerDecoderLayer(nn.Module):
    # d_model=576, nhead=12, dim_feedforward=2304, dropout_rate=0,  no_norm=false
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #     \n tgt.shape = {tgt.shape}, memory.shape = {memory.shape}, query_pos.shape = {query_pos.shape}, "))
        #  tgt.shape = torch.Size([256, 1, 576]), memory.shape = torch.Size([256, 1, 576]), query_pos.shape = torch.Size([256, 1, 576]),
        tgt2 = self.norm1(tgt)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt2.shape = {tgt2.shape},")) # tgt2.shape = torch.Size([256, 1, 576]),
        q = k = self.with_pos_embed(tgt2, query_pos)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n q.shape = {q.shape},")) # q.shape = torch.Size([256, 1, 576]),
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt2.shape = {tgt2.shape},")) # tgt2.shape = torch.Size([256, 1, 576]),
        tgt = tgt + self.dropout1(tgt2)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt.shape = {tgt.shape},")) # tgt.shape = torch.Size([256, 1, 576]),
        tgt2 = self.norm2(tgt)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt.shape = {tgt.shape},")) # tgt.shape = torch.Size([256, 1, 576]),
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory)[0]
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt2.shape = {tgt2.shape},")) #  tgt2.shape = torch.Size([256, 1, 576]),
        tgt = tgt + self.dropout2(tgt2)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt.shape = {tgt.shape},")) # tgt.shape = torch.Size([256, 1, 576]),
        tgt2 = self.norm3(tgt)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt2.shape = {tgt2.shape},")) # tgt2.shape = torch.Size([256, 1, 576]),
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n tgt2.shape = {tgt2.shape},")) # tgt2.shape = torch.Size([256, 1, 576]),
        tgt = tgt + self.dropout3(tgt2)
        #print(color.fuchsia( f"File={sys._getframe().f_code.co_filename.split('/')[-1]}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #     \n tgt.shape = {tgt.shape},")) # tgt.shape = torch.Size([256, 1, 576]),
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")





class Ipt(nn.Module):
    def __init__(self, args, ckp):
        super(Ipt, self).__init__()
        print('Making global model...')
        self.args = args
        self.scale = args.scale
        self.patch_size = args.patch_size  # 48
        self.idx_scale = 0
        self.input_large = (args.modelUse == 'VDSR')
        self.self_ensemble = args.self_ensemble  #  false
        self.precision = args.precision          #  choices=('single', 'half')
        self.cpu = args.cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
        # self.device = torch.device('cpu' if args.cpu else 'cuda:0')

        self.n_GPUs = args.n_GPUs    # 1
        self.save_models = args.save_models   # false

        self.model = ipt(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        #  /cache/results/ipt/model
        self.load(ckp.get_path('model'), resume=args.resume, cpu=args.cpu)
        print(self.model, file=ckp.log_file)

    def forward(self, x, idx_scale=0, snr=10, compressrate=0.17):
        self.idx_scale = idx_scale
        # print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x.shape = {x.shape}, idx_scale = {idx_scale}"))
        #  x.shape = torch.Size([1, 3, 256, 256]), idx_scale = 0
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        # print(f"in IPT Ipt snr = {snr}\n")

        if hasattr(self.model, 'set_snr'):
            self.model.set_snr(snr)


        if hasattr(self.model, 'set_comprate'):
            self.model.set_comprate(compressrate)

        if self.training:
            print("\nI'm in self.training.............\n")
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            #print("\nI'm not in self.training.............\n")
            forward_function = self.forward_chop

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                #print("\nhi, i'm here...\n")
                return forward_function(x)

    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(os.path.join(apath, 'model_{}.pt'.format(epoch)))

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    #  apath=/cache/results/ipt/model, resume = 0,
    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'), **kwargs)
        elif resume == 0:
            if pre_train == 'download':
                print("resume == 0\n")
                #print("pre_train................\n")
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )
        print(f"load_from = {load_from}\n")
        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

    def forward_chop(self, x, shave=12):
        x.cpu()
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n x.shape = {x.shape}"))  # x.shape = torch.Size([1, 3, 256, 256])
        batchsize = self.args.crop_batch_size  # 64

        h, w = x.size()[-2:]  # h = 256, w = 256

        padsize = int(self.patch_size)  # 48
        shave = int(self.patch_size/2)  # 24
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n h = {h}, w = {w}, padsize = {padsize}, shave = {shave}"))  # h = 256, w = 256, padsize = 48, shave = 24

        scale = self.scale[self.idx_scale]
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n scale = {scale} "))  #  scale = 2

        h_cut = (h-padsize)%(int(shave/2))   #   w_cut = 4
        w_cut = (w-padsize)%(int(shave/2))   #   h_cut = 4
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n h_cut = {h_cut}, w_cut = {w_cut} "))  # h_cut = 4, w_cut = 4

        x_unfold = torch.nn.functional.unfold(x, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_unfold.shape = {x_unfold.shape} "))  #   x_unfold.shape = torch.Size([324, 6912, 1]),
        x_hw_cut = x[...,(h-padsize):,(w-padsize):]
        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n  x_hw_cut.shape = {x_hw_cut.shape} "))
        #  x_hw_cut.shape = torch.Size([1, 3, 48, 48])

        # y_hw_cut = self.model.forward(x_hw_cut.cuda()).cpu()
        # cjj change
        y_hw_cut = self.model.forward(x_hw_cut)

        #print(color.higbluefg_whitebg( f"File={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_hw_cut.shape = {y_hw_cut.shape} "))  #  y_hw_cut.shape = torch.Size([1, 3, 96, 96])

        # 1
        x_h_cut = x[...,(h-padsize):,:]
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_h_cut.shape = {x_h_cut.shape}\n"))  #   x_h_cut.shape = torch.Size([1, 3, 48, 256])

        # 2
        x_w_cut = x[...,:,(w-padsize):]
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_w_cut.shape = {x_w_cut.shape}\n"))  #  x_w_cut.shape = torch.Size([1, 3, 256, 48])

        # x_h_cut.shape = Size([1, 3, 48, 256]), h=w=256, h_cut=4, w_cut=4, padsize=48, shave=24, scale=2, batchsize=64
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_cut.shape = {y_h_cut.shape}\n"))  #  y_h_cut.shape = torch.Size([1, 3, 96, 504])

        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_w_cut.shape = {y_w_cut.shape}\n"))  #  y_w_cut.shape = torch.Size([1, 3, 504, 96])

        # 3
        x_h_top = x[...,:padsize,:]
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_h_top.shape = {x_h_top.shape}\n"))  #   x_h_top.shape = torch.Size([1, 3, 48, 256])

        # 4
        x_w_top = x[...,:,:padsize]
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_w_top.shape = {x_w_top.shape}\n"))  #  x_w_top.shape = torch.Size([1, 3, 256, 48])


        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_top.shape = {y_h_top.shape}\n"))  #  y_h_top.shape = torch.Size([1, 3, 96, 504])

        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        # print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_w_top.shape = {y_w_top.shape}\n"))  #  y_w_top.shape = torch.Size([1, 3, 504, 96])

        x_unfold = x_unfold.view(x_unfold.size(0),-1,padsize,padsize)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_unfold.shape = {x_unfold.shape}\n"))  #  x_unfold.shape = torch.Size([324, 3, 48, 48])

        y_unfold = []

        x_range = x_unfold.size(0)//batchsize + (x_unfold.size(0)%batchsize !=0)  # 6

        if  not self.args.cpu:
            x_unfold.cuda()

        for i in range(x_range):
            if not self.args.cpu:
                if self.n_GPUs>=1:
                    y_unfold.append(P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
                else:
                    y_unfold.append(self.model(x_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
            else:
                y_unfold.append(self.model(x_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())

            # y_unfold.append(P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], ).cpu())
        y_unfold = torch.cat(y_unfold,dim=0)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n y_unfold.shape = {y_unfold.shape}\n"))  #   y_unfold.shape = torch.Size([324, 3, 96, 96])

        y = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n y.shape = {y.shape}  \n"))  #   y.shape = torch.Size([1, 3, 504, 504])

        y[...,:padsize*scale,:] = y_h_top
        y[...,:,:padsize*scale] = y_w_top

        y_unfold = y_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n y_unfold.shape = {y_unfold.shape}\n"))  #   y_unfold.shape = torch.Size([324, 3, 48, 48])

        y_inter = torch.nn.functional.fold(y_unfold.view(y_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_inter.shape = {y_inter.shape}\n"))  #   y_inter.shape = torch.Size([1, 3, 456, 456])

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_ones.shape = {y_ones.shape}\n"))  #   y_ones.shape = torch.Size([1, 3, 456, 456])

        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones, padsize*scale-shave*scale, stride=int(shave/2*scale)),((h-h_cut-shave)*scale,(w-w_cut-shave)*scale), padsize*scale-shave*scale, stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n divisor.shape = {divisor.shape}\n"))  #    divisor.shape = torch.Size([1, 3, 456, 456])

        y_inter = y_inter/divisor
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_inter.shape = {y_inter.shape}\n"))  #   y_inter.shape = torch.Size([1, 3, 456, 456])

        y[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_inter

        y = torch.cat([y[...,:y.size(2)-int((padsize-h_cut)/2*scale),:],y_h_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y.shape = {y.shape}\n"))  #   y.shape = torch.Size([1, 3, 512, 504])

        y_w_cat = torch.cat([y_w_cut[...,:y_w_cut.size(2)-int((padsize-h_cut)/2*scale),:],y_hw_cut[...,int((padsize-h_cut)/2*scale+0.5):,:]],dim=2)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_w_cat.shape = {y_w_cat.shape}\n"))  #   y_w_cat.shape = torch.Size([1, 3, 512, 96])

        y = torch.cat([y[...,:,:y.size(3)-int((padsize-w_cut)/2*scale)],y_w_cat[...,:,int((padsize-w_cut)/2*scale+0.5):]],dim=3)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y.shape = {y.shape}\n"))  #   y.shape = torch.Size([1, 3, 512, 512])

        if not self.args.cpu:
            return y.cuda()
        else:
            return y

    # x_h_cut.shape = Size([1, 3, 48, 256]), h=w=256, h_cut=4, w_cut=4, padsize=48, shave=24, scale=2, batchsize=64
    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_h_cut.shape = {x_h_cut.shape}, h={h}, w={w}, h_cut={h_cut}, w_cut={w_cut}, padsize={padsize}, shave={shave},scale={scale},batchsize={batchsize}\n"))
        #  x_h_cut.shape = torch.Size([1, 3, 48, 256]), h=256, w=256, h_cut=4, w_cut=4, padsize=48, shave=24,scale=2,batchsize=64

        x_h_cut_unfold = torch.nn.functional.unfold(x_h_cut, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_h_cut_unfold.shape = {x_h_cut_unfold.shape}\n"))  #  x_h_cut_unfold.shape = torch.Size([18, 6912, 1])

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0),-1,padsize,padsize)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_h_cut_unfold.shape = {x_h_cut_unfold.shape}\n")) #  x_h_cut_unfold.shape = torch.Size([18, 3, 48, 48])

        x_range = x_h_cut_unfold.size(0)//batchsize + (x_h_cut_unfold.size(0)%batchsize !=0)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_range  = {x_range }  \n")) #  x_range = 1
        y_h_cut_unfold=[]

        # cjj
        if not self.args.cpu:
            x_h_cut_unfold.cuda()
        else:
            pass

        for i in range(x_range):
            #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n i,batchsize = {i}, {batchsize} {x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...].shape} \n"))
            #  i,batchsize = 0, 64 torch.Size([18, 3, 48, 48])
            # y_h_cut_unfold.append(P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
            if not self.args.cpu:
                if self.n_GPUs >= 1:
                    y_h_cut_unfold.append(P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
                else:
                    y_h_cut_unfold.append(self.model(x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
            else:
                # print("I'am use CPU...\n")
                y_h_cut_unfold.append(self.model(x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())

        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n len(y_h_cut_unfold)  = {len(y_h_cut_unfold)}, y_h_cut_unfold[0].shape = {y_h_cut_unfold[0].shape}  \n")) #  x_range = 1
        #  len(y_h_cut_unfold)  = 1, y_h_cut_unfold[0].shape = torch.Size([18, 3, 96, 96])

        y_h_cut_unfold = torch.cat(y_h_cut_unfold,dim=0)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_cut_unfold = {y_h_cut_unfold.shape} \n"))
        #   y_h_cut_unfold = torch.Size([18, 3, 96, 96])

        y_h_cut = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut)*scale), padsize*scale, stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_cut.shape = {y_h_cut.shape}, \n"))
        #   y_h_cut.shape = torch.Size([1, 3, 96, 504]), y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).shape = torch.Size([18, 27648, 1])

        y_h_cut_unfold = y_h_cut_unfold[...,:,int(shave/2*scale):padsize*scale-int(shave/2*scale)].contiguous()
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_cut_unfold.shape = {y_h_cut_unfold.shape} \n"))
        #   y_h_cut_unfold.shape = torch.Size([18, 3, 96, 48])

        y_h_cut_inter = torch.nn.functional.fold(y_h_cut_unfold.view(y_h_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale))
        # print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_cut_inter.shape = {y_h_cut_inter.shape} \n"))#  y_h_cut_inter.shape = torch.Size([1, 3, 96, 456])

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)  #  torch.Size([1, 3, 96, 456])
        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones ,(padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale)),(padsize*scale,(w-w_cut-shave)*scale), (padsize*scale,padsize*scale-shave*scale), stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n divisor.shape = {divisor.shape} \n"))  # divisor.shape = torch.Size([1, 3, 96, 456])

        y_h_cut_inter = y_h_cut_inter/divisor
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_h_cut_inter.shape = {y_h_cut_inter.shape} \n"))  # divisor.shape = torch.Size([1, 3, 96, 456])

        y_h_cut[...,:,int(shave/2*scale):(w-w_cut)*scale-int(shave/2*scale)] = y_h_cut_inter
        return y_h_cut

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_w_cut.shape = {x_w_cut.shape}, h={h}, w={w}, h_cut={h_cut}, w_cut={w_cut}, padsize={padsize}, shave={shave},scale={scale},batchsize={batchsize}\n"))
        #   x_w_cut.shape = torch.Size([1, 3, 256, 48]), h=256, w=256, h_cut=4, w_cut=4, padsize=48, shave=24,scale=2,batchsize=64

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave/2)).transpose(0,2).contiguous()
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_w_cut_unfold.shape = {x_w_cut_unfold.shape}\n"))  #   x_w_cut_unfold.shape = torch.Size([18, 6912, 1])

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0),-1,padsize,padsize)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_w_cut_unfold.shape = {x_w_cut_unfold.shape}\n"))  #   x_w_cut_unfold.shape = torch.Size([18, 3, 48, 48])

        x_range = x_w_cut_unfold.size(0)//batchsize + (x_w_cut_unfold.size(0)%batchsize !=0)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n x_range = {x_range}\n"))  #   x_range = 1

        y_w_cut_unfold=[]

        if not self.args.cpu:
            x_w_cut_unfold.cuda()
        else:
            pass

        for i in range(x_range):
            if not self.args.cpu:
                if self.n_GPUs >= 1:
                    y_w_cut_unfold.append(P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
                else:
                    y_w_cut_unfold.append(self.model(x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
            else:
                y_w_cut_unfold.append(self.model(x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...]).cpu())
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #  \n len(y_w_cut_unfold) = {len(y_w_cut_unfold)}\n"))  #  len(y_w_cut_unfold) = 1

        y_w_cut_unfold = torch.cat(y_w_cut_unfold,dim=0)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\
        #      \n y_w_cut_unfold.shape = {y_w_cut_unfold.shape}\n"))  #   y_w_cut_unfold.shape = torch.Size([18, 3, 96, 96])

        y_w_cut = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut)*scale,padsize*scale), padsize*scale, stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_w_cut.shape = {y_w_cut.shape}\n"))  #   y_w_cut.shape = torch.Size([1, 3, 504, 96])

        y_w_cut_unfold = y_w_cut_unfold[...,int(shave/2*scale):padsize*scale-int(shave/2*scale),:].contiguous()
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_w_cut_unfold.shape = {y_w_cut_unfold.shape}\n"))  #  y_w_cut_unfold.shape = torch.Size([18, 3, 48, 96])

        y_w_cut_inter = torch.nn.functional.fold(y_w_cut_unfold.view(y_w_cut_unfold.size(0),-1,1).transpose(0,2).contiguous(),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_w_cut_inter.shape = {y_w_cut_inter.shape}\n"))  # y_w_cut_inter.shape = torch.Size([1, 3, 456, 96])


        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n y_ones.shape = {y_ones.shape}\n"))  # y_ones.shape = torch.Size([1, 3, 456, 96])

        divisor = torch.nn.functional.fold(torch.nn.functional.unfold(y_ones ,(padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale)),((h-h_cut-shave)*scale,padsize*scale), (padsize*scale-shave*scale,padsize*scale), stride=int(shave/2*scale))
        #print(color.higbluefg_whitebg( f"\nFile={'/'.join(sys._getframe().f_code.co_filename.split('/')[-2:])}, Func={sys._getframe().f_code.co_name}, Line={sys._getframe().f_lineno}\n divisor.shape = {divisor.shape}\n"))  #   divisor.shape = torch.Size([1, 3, 456, 96])

        y_w_cut_inter = y_w_cut_inter/divisor

        y_w_cut[...,int(shave/2*scale):(h-h_cut)*scale-int(shave/2*scale),:] = y_w_cut_inter
        return y_w_cut




