from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import models.configs as configs
from models.attention import Attention
from models.embed import Embeddings 
from models.mlp import Mlp

class FeaturePooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1) 
        return (x * attn_weights).sum(dim=1) 
     

class Block(nn.Module):
    def __init__(self, config, vis, mm=False):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        if mm:
            self.att_norm_text = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm_text = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_text = Mlp(config)
            
            self.att_norm_clinical = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_norm_clinical = LayerNorm(config.hidden_size, eps=1e-6)
            self.ffn_clinical = Mlp(config)
            
            self.gating_mlp = nn.Sequential(
                nn.Linear(3 * config.hidden_size, 128), 
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 3), 
                nn.Softmax(dim=-1)
            )
            for layer in self.gating_mlp:
                if isinstance(layer, nn.Linear): 
                    if layer.out_features == 3:  
                        nn.init.uniform_(layer.weight, -0.01, 0.01)
                        nn.init.constant_(layer.bias, 0.0)
                    else:
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

        self.use_gating = config.use_gating
        self.pooler_img = FeaturePooler(config.hidden_size)
        self.pooler_text = FeaturePooler(config.hidden_size)
        self.pooler_clinical = FeaturePooler(config.hidden_size)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis, mm)

    def forward(self, x, text=None, clinical=None):
        if text is None and clinical is None:            
            h = x
            x = self.attention_norm(x)
            x, weights = self.attn(x)
            x = x + h

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x, weights
        else:            
            h = x
            h_text = text
            h_clinical = clinical
            
            x = self.attention_norm(x)
            text = self.att_norm_text(text)
            clinical = self.att_norm_clinical(clinical)
            
            x, text, clinical, weights = self.attn(x, text, clinical)
            x = x + h
            text = text + h_text
            clinical = clinical + h_clinical
            
            h = x
            h_text = text
            h_clinical = clinical

            x = self.ffn_norm(x)
            text = self.ffn_norm_text(text)
            clinical = self.ffn_norm_clinical(clinical)

            x = self.ffn(x)
            text = self.ffn_text(text)
            clinical = self.ffn_clinical(clinical)

            x = x + h
            text = text + h_text
            clinical = clinical + h_clinical

            if self.use_gating:
                
                pooled_img = self.pooler_img(x)
                pooled_text = self.pooler_text(text)
                pooled_clinical = self.pooler_clinical(clinical)
                
                combined = torch.cat([pooled_img, pooled_text, pooled_clinical], dim=-1) 
                
                modality_weights = self.gating_mlp(combined) 
                w_img, w_text, w_clinical = modality_weights[:, 0], modality_weights[:, 1], modality_weights[:, 2]
                
                x = w_img.unsqueeze(-1).unsqueeze(-1) * x + x  
                text = w_text.unsqueeze(-1).unsqueeze(-1) * text + text 
                clinical = w_clinical.unsqueeze(-1).unsqueeze(-1) * clinical + clinical  

                return x, text, clinical, weights
            else:
                return x, text, clinical, weights
