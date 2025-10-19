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
from models.encoder_parallel import EncoderParallel
import pdb

logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = EncoderParallel(config, vis)

    def forward(self, input_ids, cc=None, lab=None, sex=None, age=None, cohort=None):
        embedding_output, cc, lab, sex, age, cohort = self.embeddings(input_ids, cc, lab, sex, age, cohort)
        text = cc
        clinical = torch.cat((lab, sex, age), 1)
        encoded, attn_weights = self.encoder(embedding_output, text, clinical, cohort)
        return encoded, attn_weights


class TRIM(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1, zero_head=False, vis=False):
        super(TRIM, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, cc=None, lab=None, sex=None, age=None, cohort=None, labels=None):
        x, attn_weights = self.transformer(x, cc, lab, sex, age, cohort)
        logits = self.head(torch.mean(x, dim=1)) 
        return logits.squeeze(), attn_weights 
