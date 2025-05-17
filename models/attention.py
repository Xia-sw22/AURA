import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import models.configs as configs
import math

class Attention(nn.Module):
    def __init__(self, config, vis, mm=True):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        if mm:
            self.query_text = Linear(config.hidden_size, self.all_head_size)
            self.key_text = Linear(config.hidden_size, self.all_head_size)
            self.value_text = Linear(config.hidden_size, self.all_head_size)
            self.out_text = Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout_text = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_it = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_ti = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout_text = Dropout(config.transformer["attention_dropout_rate"])

            self.query_clinical = Linear(config.hidden_size, self.all_head_size)
            self.key_clinical = Linear(config.hidden_size, self.all_head_size)
            self.value_clinical = Linear(config.hidden_size, self.all_head_size)
            self.out_clinical = Linear(config.hidden_size, config.hidden_size)
            self.attn_dropout_clinical = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_ic = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_ci = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_tc = Dropout(config.transformer["attention_dropout_rate"])
            self.attn_dropout_ct = Dropout(config.transformer["attention_dropout_rate"])
            self.proj_dropout_clinical = Dropout(config.transformer["attention_dropout_rate"])

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, text=None, clinical=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if text is not None:
            text_q = self.query_text(text)      
            text_k = self.key_text(text)      
            text_v = self.value_text(text)      

        if clinical is not None:
            clinical_q = self.query_clinical(clinical)
            clinical_k = self.key_clinical(clinical)
            clinical_v = self.value_clinical(clinical)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if text is not None:
            query_layer_img = query_layer
            key_layer_img = key_layer
            value_layer_img = value_layer
            query_layer_text = self.transpose_for_scores(text_q)
            key_layer_text = self.transpose_for_scores(text_k)
            value_layer_text = self.transpose_for_scores(text_v)

        if clinical is not None:
            query_layer_clinical = self.transpose_for_scores(clinical_q)
            key_layer_clinical = self.transpose_for_scores(clinical_k)
            value_layer_clinical = self.transpose_for_scores(clinical_v)

        if text is None and clinical is None:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None
            attention_probs = self.attn_dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)
            attention_output = self.proj_dropout(attention_output)
            return attention_output, weights
        else:
            attention_scores_img = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores_text = torch.matmul(query_layer_text, key_layer_text.transpose(-1, -2))
            attention_scores_clinical = torch.matmul(query_layer_clinical, key_layer_clinical.transpose(-1, -2))
            attention_scores_it = torch.matmul(query_layer_img, key_layer_text.transpose(-1, -2))
            attention_scores_ti = torch.matmul(query_layer_text, key_layer_img.transpose(-1, -2))
            attention_scores_ic = torch.matmul(query_layer_img, key_layer_clinical.transpose(-1, -2))
            attention_scores_ci = torch.matmul(query_layer_clinical, key_layer_img.transpose(-1, -2))
            attention_scores_tc = torch.matmul(query_layer_text, key_layer_clinical.transpose(-1, -2))
            attention_scores_ct = torch.matmul(query_layer_clinical, key_layer_text.transpose(-1, -2))

            attention_scores_img = attention_scores_img / math.sqrt(self.attention_head_size)
            attention_probs_img = self.softmax(attention_scores_img)
            weights = attention_probs_img if self.vis else None
            attention_probs_img = self.attn_dropout(attention_probs_img)

            attention_scores_text = attention_scores_text / math.sqrt(self.attention_head_size)
            attention_probs_text = self.softmax(attention_scores_text)
            attention_probs_text = self.attn_dropout_text(attention_probs_text)

            attention_scores_clinical = attention_scores_clinical / math.sqrt(self.attention_head_size)
            attention_probs_clinical = self.softmax(attention_scores_clinical)
            attention_probs_clinical = self.attn_dropout_clinical(attention_probs_clinical)

            attention_scores_it = attention_scores_it / math.sqrt(self.attention_head_size)
            attention_probs_it = self.softmax(attention_scores_it)
            attention_probs_it = self.attn_dropout_it(attention_probs_it)

            attention_scores_ti = attention_scores_ti / math.sqrt(self.attention_head_size)
            attention_probs_ti = self.softmax(attention_scores_ti)
            attention_probs_ti = self.attn_dropout_ti(attention_probs_ti)

            attention_scores_ic = attention_scores_ic / math.sqrt(self.attention_head_size)
            attention_probs_ic = self.softmax(attention_scores_ic)
            attention_probs_ic = self.attn_dropout_ic(attention_probs_ic)

            attention_scores_ci = attention_scores_ci / math.sqrt(self.attention_head_size)
            attention_probs_ci = self.softmax(attention_scores_ci)
            attention_probs_ci = self.attn_dropout_ci(attention_probs_ci)

            attention_scores_tc = attention_scores_tc / math.sqrt(self.attention_head_size)
            attention_probs_tc = self.softmax(attention_scores_tc)
            attention_probs_tc = self.attn_dropout_tc(attention_probs_tc)

            attention_scores_ct = attention_scores_ct / math.sqrt(self.attention_head_size)
            attention_probs_ct = self.softmax(attention_scores_ct)
            attention_probs_ct = self.attn_dropout_ct(attention_probs_ct)

            context_layer_img = torch.matmul(attention_probs_img, value_layer_img)
            context_layer_img = context_layer_img.permute(0, 2, 1, 3).contiguous()
            context_layer_text = torch.matmul(attention_probs_text, value_layer_text)
            context_layer_text = context_layer_text.permute(0, 2, 1, 3).contiguous()
            context_layer_clinical = torch.matmul(attention_probs_clinical, value_layer_clinical)
            context_layer_clinical = context_layer_clinical.permute(0, 2, 1, 3).contiguous()
            context_layer_it = torch.matmul(attention_probs_it, value_layer_text)
            context_layer_it = context_layer_it.permute(0, 2, 1, 3).contiguous()
            context_layer_ti = torch.matmul(attention_probs_ti, value_layer_img)
            context_layer_ti = context_layer_ti.permute(0, 2, 1, 3).contiguous()
            context_layer_ic = torch.matmul(attention_probs_ic, value_layer_clinical)
            context_layer_ic = context_layer_ic.permute(0, 2, 1, 3).contiguous()
            context_layer_ci = torch.matmul(attention_probs_ci, value_layer_img)
            context_layer_ci = context_layer_ci.permute(0, 2, 1, 3).contiguous()
            context_layer_tc = torch.matmul(attention_probs_tc, value_layer_clinical)
            context_layer_tc = context_layer_tc.permute(0, 2, 1, 3).contiguous()
            context_layer_ct = torch.matmul(attention_probs_ct, value_layer_text)
            context_layer_ct = context_layer_ct.permute(0, 2, 1, 3).contiguous()

            new_context_layer_shape = context_layer_img.size()[:-2] + (self.all_head_size,)
            context_layer_img = context_layer_img.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_text.size()[:-2] + (self.all_head_size,)
            context_layer_text = context_layer_text.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_clinical.size()[:-2] + (self.all_head_size,)
            context_layer_clinical = context_layer_clinical.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_it.size()[:-2] + (self.all_head_size,)
            context_layer_it = context_layer_it.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_ti.size()[:-2] + (self.all_head_size,)
            context_layer_ti = context_layer_ti.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_ic.size()[:-2] + (self.all_head_size,)
            context_layer_ic = context_layer_ic.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_ci.size()[:-2] + (self.all_head_size,)
            context_layer_ci = context_layer_ci.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_tc.size()[:-2] + (self.all_head_size,)
            context_layer_tc = context_layer_tc.view(*new_context_layer_shape)
            new_context_layer_shape = context_layer_ct.size()[:-2] + (self.all_head_size,)
            context_layer_ct = context_layer_ct.view(*new_context_layer_shape)

            attention_output_img = self.out((context_layer_img + context_layer_it + context_layer_ic) / 3)
            attention_output_text = self.out((context_layer_text + context_layer_ti + context_layer_tc) / 3)
            attention_output_clinical = self.out((context_layer_clinical + context_layer_ci + context_layer_ct) / 3)
            attention_output_img = self.proj_dropout(attention_output_img)
            attention_output_text = self.proj_dropout_text(attention_output_text)
            attention_output_clinical = self.proj_dropout_clinical(attention_output_clinical)

            return attention_output_img, attention_output_text, attention_output_clinical, weights