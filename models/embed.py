# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Conv2d
from torch.nn.modules.utils import _pair

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.cc_len
        num_lab = config.lab_len

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.img_embeddings = Linear(768, config.hidden_size)
        self.cc_embeddings = Linear(768, config.hidden_size)  
        self.lab_embeddings = Linear(1, config.hidden_size)  
        self.sex_embeddings = Linear(1, config.hidden_size)  
        self.age_embeddings = Linear(1, config.hidden_size)  
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + n_patches, config.hidden_size))
        self.pe_img = nn.Parameter(torch.zeros(1, 197, config.hidden_size))
        self.pe_cc = nn.Parameter(torch.zeros(1, tk_lim, config.hidden_size))
        self.pe_lab = nn.Parameter(torch.zeros(1, num_lab, config.hidden_size))
        self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_cc = Dropout(config.transformer["dropout_rate"])
        self.dropout_lab = Dropout(config.transformer["dropout_rate"])
        self.dropout_sex = Dropout(config.transformer["dropout_rate"])
        self.dropout_age = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, cc, lab, sex, age):
        B = x.shape[0]
        if x.dim() == 4:  
            S = x.shape[1]
            x = x.view(B * S, *x.shape[2:])  
            x = self.img_embeddings(x)    
            x = x.view(B, S, *x.shape[1:]) 
            x = x.mean(dim=1)        
        else:  
            x = self.img_embeddings(x)
        
        cc = self.cc_embeddings(cc)
        lab = self.lab_embeddings(lab)
        sex = self.sex_embeddings(sex)
        age = self.age_embeddings(age)
        
        embeddings = x + self.pe_img
        cc_embeddings = cc + self.pe_cc
        lab_embeddings = lab + self.pe_lab
        sex_embeddings = sex + self.pe_sex
        age_embeddings = age + self.pe_age

        embeddings = self.dropout(embeddings)
        cc_embeddings = self.dropout_cc(cc_embeddings)
        lab_embeddings = self.dropout_lab(lab_embeddings)
        sex_embeddings = self.dropout_sex(sex_embeddings)
        age_embeddings = self.dropout_age(age_embeddings)

        return embeddings, cc_embeddings, lab_embeddings, sex_embeddings, age_embeddings