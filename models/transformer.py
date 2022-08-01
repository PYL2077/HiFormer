import copy
from tkinter import N
from turtle import forward
from typing import Optional
# from turtle import forward
import torch
from torch import nn, Tensor
class TransformerEncoder(nn.Module):
    def __init__(self, dim_model=512, nhead=8,
                 num_enc_layers=6, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_verb_classes = 504
        self.verb_tokens = nn.Parameter(torch.zeros((1,dim_model)), requires_grad = True)
        # Encoder
        enc_layer = Encoder_enc_Layer(dim_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder_enc(enc_layer, num_enc_layers)
        
        # Verb Classifier
        self.verb_classifier = nn.Sequential(nn.Linear(dim_model, dim_model),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(dim_model, self.num_verb_classes))
        self.norm1 = nn.LayerNorm(dim_model)
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
    def forward(self, img_ft, mask,
                pos_embed, vidx_ridx,
                targets=None, inference=False):
        assert img_ft.shape[1] == self.dim_model
        assert img_ft.shape == pos_embed.shape
        device = img_ft.device

        # flatten from N,C,H,W to HW,N,C
        bs,c,h,w = img_ft.shape
        img_ft = img_ft.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        # self.verb_tokens.to(device)

        # Encoder first half
        src = torch.cat((self.verb_tokens.unsqueeze(1).repeat(1, bs, 1), img_ft), dim=0)
        enc_zero_mask = torch.zeros((bs, 1), dtype=torch.bool, device=device)
        mem_mask = torch.cat((enc_zero_mask, mask), dim=1)
        verb_ft, img_ft = self.encoder(src, src_key_padding_mask=mem_mask, pos=pos_embed, num_zeros = 1).split([1, h*w], dim=0)

        # verb prediction
        verb_ft = verb_ft.view(bs, -1)
        verb_pred = self.verb_classifier(self.norm1(verb_ft)).view(bs, self.num_verb_classes)
        
        return img_ft, verb_ft, verb_pred

class Encoder_enc(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, num_zeros=num_zeros)
        return src
class Encoder_dec(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
        return tgt
class Encoder_enc_Layer(nn.Module):
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
        src2 = self.mha(src, src, src2, attn_mask=src_mask,
                       key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        return self.norm3(src)

class Encoder_dec_Layer(nn.Module):
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
    def pos_embed(self, t, pos, num_zeros=None):
        if num_zeros is None: 
            return t+pos
        else: 
            return t if pos is None else torch.cat((t[:num_zeros],(t[num_zeros:]+pos)),dim=0)
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt = self.norm1(tgt)
        q = tgt
        k = v = self.pos_embed(memory, pos)
        tgt2 = self.mha(q, k, v,
                        attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout2(tgt2)
        return self.norm3(tgt)

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
def build_transformer_encoder(args):
    return TransformerEncoder(dim_model=args.hidden_dim,
                   dropout=args.dropout,
                   nhead=args.nheads,
                   num_enc_layers=args.num_enc_layers,
                   dim_feedforward=args.dim_feedforward)