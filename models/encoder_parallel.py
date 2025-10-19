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
from models.block import Block

class ParallelCrossAttentionChannel(nn.Module):
    """单个交叉注意力通道，包含两层交叉注意力层和门控机制"""
    def __init__(self, config, vis):
        super(ParallelCrossAttentionChannel, self).__init__()
        self.vis = vis
        self.config = config
        
        self.cross_attn_layer1 = Block(config, vis, mm=True)
        self.cross_attn_layer2 = Block(config, vis, mm=True)
        
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
    
    def forward(self, img_features, text_features, clinical_features):
        """前向传播，处理单个队列的样本"""
        img_out1, text_out1, clinical_out1, weights1 = self.cross_attn_layer1(
            img_features, text_features, clinical_features
        )
        
        img_out2, text_out2, clinical_out2, weights2 = self.cross_attn_layer2(
            img_out1, text_out1, clinical_out1
        )
        
        if self.config.use_gating:
            pooled_img = torch.mean(img_out2, dim=1)
            pooled_text = torch.mean(text_out2, dim=1)
            pooled_clinical = torch.mean(clinical_out2, dim=1)
            
            combined = torch.cat([pooled_img, pooled_text, pooled_clinical], dim=-1)
            modality_weights = self.gating_mlp(combined)
            w_img, w_text, w_clinical = modality_weights[:, 0], modality_weights[:, 1], modality_weights[:, 2]
            
            img_out2 = w_img.unsqueeze(-1).unsqueeze(-1) * img_out2 + img_out2
            text_out2 = w_text.unsqueeze(-1).unsqueeze(-1) * text_out2 + text_out2
            clinical_out2 = w_clinical.unsqueeze(-1).unsqueeze(-1) * clinical_out2 + clinical_out2
        
        combined_features = torch.cat([img_out2, text_out2, clinical_out2], dim=1)
        
        return combined_features, [weights1, weights2]

class EncoderParallel(nn.Module):
    """支持并行通道的编码器"""
    def __init__(self, config, vis):
        super(EncoderParallel, self).__init__()
        self.vis = vis
        self.config = config
        
        self.ets_channel = ParallelCrossAttentionChannel(config, vis)
        self.es_channel = ParallelCrossAttentionChannel(config, vis)
        
        self.shared_layers = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        for i in range(config.transformer["num_layers"] - 2):
            layer = Block(config, vis, mm=False)
            self.shared_layers.append(copy.deepcopy(layer))
    
    def forward(self, hidden_states, text=None, clinical=None, cohort=None):
        """
        前向传播
        Args:
            hidden_states: 图像特征
            text: 文本特征
            clinical: 临床特征
            cohort: 队列标识 (0: ETS, 1: ES)
        """
        attn_weights = []
        
        if cohort is not None:
            ets_mask = (cohort == 0)
            es_mask = (cohort == 1)
            
            batch_size = hidden_states.shape[0]
            total_seq_len = hidden_states.shape[1] + text.shape[1] + clinical.shape[1]
            combined_features = torch.zeros(batch_size, total_seq_len, self.config.hidden_size, 
                                          device=hidden_states.device, dtype=hidden_states.dtype)
            
            if ets_mask.any():
                ets_img = hidden_states[ets_mask]
                ets_text = text[ets_mask]
                ets_clinical = clinical[ets_mask]
                
                ets_combined, ets_weights = self.ets_channel(ets_img, ets_text, ets_clinical)
                combined_features[ets_mask] = ets_combined
                if self.vis:
                    attn_weights.extend(ets_weights)
            
            if es_mask.any():
                es_img = hidden_states[es_mask]
                es_text = text[es_mask]
                es_clinical = clinical[es_mask]
                
                es_combined, es_weights = self.es_channel(es_img, es_text, es_clinical)
                combined_features[es_mask] = es_combined
                if self.vis:
                    attn_weights.extend(es_weights)
        else:
            combined_features, channel_weights = self.ets_channel(hidden_states, text, clinical)
            if self.vis:
                attn_weights.extend(channel_weights)
        
        for layer_block in self.shared_layers:
            combined_features, weights = layer_block(combined_features)
            if self.vis:
                attn_weights.append(weights)
        
        encoded = self.encoder_norm(combined_features)
        return encoded, attn_weights
