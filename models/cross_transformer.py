import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .layers import MultiHeadAttention


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

class CrossTransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(CrossTransformerLayer, self).__init__()
        
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn_x = MultiHeadAttention(d_model, nhead, attn_dropout)
        self.self_attn_y = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1_x = nn.Linear(d_model, dim_feedforward)
        self.linear1_y = nn.Linear(d_model, dim_feedforward)
        self.dropout_x = nn.Dropout(act_dropout)
        self.dropout_y = nn.Dropout(act_dropout)
        self.linear2_x = nn.Linear(dim_feedforward, d_model)
        self.linear2_y = nn.Linear(dim_feedforward, d_model)

        self.norm1_x = nn.LayerNorm(d_model)
        self.norm1_y = nn.LayerNorm(d_model)
        
        
        self.norm2_x = nn.LayerNorm(d_model)
        self.norm2_y = nn.LayerNorm(d_model)
        
        self.dropout1_x = nn.Dropout(dropout)
        self.dropout1_y = nn.Dropout(dropout)
        
        self.dropout2_x = nn.Dropout(dropout)
        self.dropout2_y = nn.Dropout(dropout)
        
        self.activation = _get_activation_fn(activation)


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, y, x_mask=None, y_mask=None, pos=None):
        x_residual = x
        y_residual = y
        
        if self.normalize_before:
            x = self.norm1_x(x)
            y = self.norm1_y(y)
        
        
        x_q = x_k = self.with_pos_embed(x, pos)
        y_q = y_k = self.with_pos_embed(y, pos)
        # x_q
        # 计算交叉注意力
        # q:x, k:y, v:y 
        
        x = self.self_attn_x(x_q, y_k, value=y, attn_mask=x_mask)
        
        # q:y, k:x, v:x
        y = self.self_attn_y(y_q, x_k, value=x, attn_mask=y_mask)
        
        # norm+残差
        x = x_residual + self.dropout1_x(x)
        if not self.normalize_before:
            x = self.norm1_x(x)
        
        y = y_residual + self.dropout1_y(y)
        if not self.normalize_before:
            y = self.norm1_y(y)

        # FFN +LN
        x_residual = x
        y_residual = y
        
        if self.normalize_before:
            x = self.norm2_x(x)
            y = self.norm2_y(y)
        
        x = self.linear2_x(self.dropout_x(self.activation(self.linear1_x(x))))
        x = x_residual + self.dropout2_x(x)
        if not self.normalize_before:
            x = self.norm2_x(x)

        y = self.linear2_y(self.dropout_y(self.activation(self.linear1_y(y))))
        y = y_residual + self.dropout2_y(y)
        if not self.normalize_before:
            y = self.norm2_y(y)
         
        return x, y


class CrossTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x,y,
                x_mask: Optional[Tensor] = None,
                y_mask: Optional[Tensor] = None,
                x_key_padding_mask: Optional[Tensor] = None,
                y_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        output_x = x
        output_y = y

        for layer in self.layers:
            output_x, output_y = layer(x, y, x_mask=x_mask, y_mask=y_mask, pos=pos)

        if self.norm is not None:
            output_x = self.norm(output_x)
            output_y = self.norm(output_y)

        return output_x, output_y