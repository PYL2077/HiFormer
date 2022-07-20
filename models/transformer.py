import copy
from tkinter import N
from turtle import forward
from typing import Optional
# from turtle import forward
import torch
from torch import nn, Tensor
class Encoder(nn.Module):
    def __init__(self, dim_model=512, nhead=8,
                 num_enc_layers=6, dim_feedforward=2048):
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_verb_classes = 504
        # Encoder
        self.enc_layer_A = EncoderLayer_A(dim_model, nhead, dim_feedforward=dim_feedforward)
        self.enc_layer_B = EncoderLayer_B(dim_model, nhead, dim_feedforward=dim_feedforward)
        self.enc_A = Encoder_A(self.enc_layer_A, num_enc_layers)
        self.enc_B = Encoder_B(self.enc_layer_B, num_enc_layers)
        
        # Verb Classifier
        self.verb_classifier = nn.Sequential(nn.Linear(dim_model*2, dim_model*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(dim_model*2, self.num_verb_classes))
        self.norm1 = nn.LayerNorm(dim_model*2)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1: 
                nn.init.xavier_uniform_(p)
    def pos_embed(self, t, pos, num_zeros=None):
        if num_zeros is None: 
            return t+pos
        else: 
            return t if pos is None else torch.cat(t[:num_zeros],(t[num_zeros:]+pos),dim=0)
    def forward(self, img_ft, verb_ft, mask,
                IL_token_embed, RL_token_embed,
                verb_token_embed, role_token_embed,
                pos_embed, vidx_ridx,
                targets=None, inference=False):
        assert img_ft.shape[1] == self.dim_model
        assert verb_ft.shape == self.dim_model
        assert img_ft.shape == pos_embed.shape

        device = img_ft.device

        # flatten from N,C,H,W to HW,N,C
        bs,c,h,w = img_ft.shape
        img_ft = img_ft.flatten(2).permute(2, 0, 1)
        verb_ft = verb_ft.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        # Encoder first half
        src = torch.cat((img_ft,verb_ft),dim=0)
        img_ft, verb_ft = self.enc_A(src, src_key_padding_mask=mask, pos=pos_embed).split(1)

        # verb prediction
        verb_ft = self.norm1(verb_ft)
        verb_pred = self.verb_classifier(verb_ft)
        
        if not inference:
            selected_roles = targets["roles"]
        else:
            top1_verb = torch.topk(verb_pred, k=1, dim=1)[1].item()
            selected_roles = vidx_ridx[top1_verb]
        # Encoder second half
        mem = self.pos_embed(img_ft, pos_embed)
        src = torch.cat((verb_ft, selected_roles), dim=0)
        src = self.enc_B(src, mem)
        verb_ft, selected_roles = src.split(self.dim_model)
        return verb_ft, selected_roles
class Transformer(nn.Module):
    def __init__(self, dim_model=512, nhead=8,
                 num_enc_layers=6, num_dec_layers=5, dim_feedforward=2048):
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_verb_classes = 504
        # Encoder
        enc_layer = TransformerEncoderLayer(dim_model, nhead, dim_feedforward=dim_feedforward)
        self.enc_1 = TransformerEncoder(enc_layer, num_enc_layers)
        self.enc_2 = TransformerEncoder(enc_layer, num_enc_layers)
        # Decoder

        # Verb_Classifier
        self.verb_classifier = nn.Sequential(nn.Linear(dim_model*2, dim_model*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(dim_model*2, self.num_verb_classes))
        self.norm1 = nn.LayerNorm(dim_model*2)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim()>1: 
                nn.init.xavier_uniform_(p)
    def forward(self, img_ft, verb_ft, mask, pos_embed,
                targets=None, inference=False):
        """
        mask is allways all-zero?
        Probably not during inference
        """

        device = img_ft.device
        """
        # flatten N,C,H,W to HW,N,C
        bs,c,h,w = src.shape
        src.flatten(2).permute(2,0,1)
        pos_embed.flatten(2).permute(2,0,1)
        mask.flatten(1)
        """
        # Encoder
        # Encoder 1
        src = torch.cat((verb_ft, img_ft), dim=0)
        verb_ft, img_ft = self.enc_1(src, src_key_padding_mask=mask, pos=pos_embed).split(1)
        # verb classifier
        role_embed = self.verb_classifier(verb_ft)
        # Encoder 2

        # Decoder
        # now we have src as [hv, h_v_n_1~m]
        h_verb, h_noun = src.split(1)
        n = 5 # support verb size
class Encoder_A(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
    def forward(self, src,
                src_key_padding_mask = None,
                pos = None, num_zeros = None):
        for layer in self.layers:
            src = layer(src,
                        src_key_padding_mask=src_key_padding_mask,
                        pos=pos, num_zeros=num_zeros)
        return src
class Encoder_B(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
    def forward(self, src, memory):
        for layer in self.layers:
            src = layer(src, memory)
        return src
class EncoderLayer_A(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.ffn = MLP(dim_model)
        # self.activation = _get_activation(activation)
        self.ffn = nn.Sequential(nn.Linear(dim_model, dim_feedforward),
                                 _get_activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, dim_model))
    def pos_embed(self, t, pos, num_zeros=None):
        if num_zeros is None: 
            return t+pos
        else: 
            return t if pos is None else torch.cat((t[:num_zeros],(t[num_zeros:]+pos)),dim=0)
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros=None):
        src2 = self.norm1(src)
        src = self.pos_embed(src2, pos=pos, num_zeros=num_zeros)
        # MHA block
        src2 = self.mha(src, src, src2, attn_mask=src_mask,
                       key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        return self.norm3(src)
class EncoderLayer_B(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.ffn = MLP(dim_model)
        self.ffn = nn.Sequential(nn.Linear(dim_model, dim_feedforward),
                                 _get_activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, dim_model))
        self.activation = _get_activation(activation)
    def forward(self, src, memory, src_key_padding_mask=None):
        src = self.norm1(src)
        src2 = self.mha(src, memory, memory, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.ffn(src)
        src = src + self.dropout1(src2)
        return self.norm3(src)
class MLP(nn.Module):
    """ Feedforward Network """
    def __init__(self, dim_model, hidden_dim=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(dim_model,hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim,dim_model)
        self.activation = _get_activation(activation)

    def forward(self, src):
        src = self.linear1(src)
        src = self.activation(src)
        src = self.dropout(src)
        src = self.linear2(src)
        return src
def _get_activation(opt):
    if opt == "relu":
        return nn.ReLU()
    elif opt == "gelu":
        return nn.GELU()
    elif opt == "glu":
        return nn.GLU()
    else: raise Exception("Unexpected activation type")
def _get_clones(layer, n):
    return nn.ModuleList(copy.deepcopy(layer) for _ in range(n))
