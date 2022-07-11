import copy
from typing import Optional
# from turtle import forward
import torch
from torch import nn, Tensor
class Transformer(nn.Module):
    def __init__(self, dim_model=512, nhead=8,
                 num_enc_layers=6, num_dec_layers=5, dim_ffn=2048):
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_verb_classes = 504
        # Encoder
        enc_1_layer = TransformerEncoderLayer(dim_model, nhead, dim_ffn=dim_ffn)
        self.enc_1 = TransformerEncoder(enc_1_layer, num_enc_layers)
        self.enc_2 = TransformerEncoder(enc_layer, num_enc_layers)
        # Decoder

        # Verb_Classifier
        self.verb_classifier = nn.Sequential(nn.Linear(dim_model*2,dim_model*2),
                                             nn.ReLU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(dim_model*2,self.num_verb_classes))
        self.ln1 = nn.LayerNorm(dim_model*2)
        self._init_weight()
        
    def _init_weight(self):
        for p in self.parameters():
            if p.dim()>1: 
                nn.init.xavier_uniform_(p)
    def forward(self, src, mask,
                verb_token_embed, noun_token_embed, pos_embed,
                targets=None, inference=False):
        device = src.device
        # flatten N,C,H,W to HW,N,C
        bs,c,h,w = src.shape
        src.flatten(2).permute(2,0,1)
        pos_embed.flatten(2).permute(2,0,1)
        mask.flatten(1)
        # Encoder
        

        
class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros = None):
        for layer in self.layers:
            src = layer(src, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask,
                        pos=pos, num_zeros=num_zeros)
        return src
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, nhead, dim_ffn=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim_model)
        self.ln2 = nn.LayerNorm(dim_model)
        self.ln3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.ffn = MLP(dim_model)
        self.ffn = nn.Sequential(nn.Linear(dim_model, dim_ffn),
                                 _get_activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_ffn, dim_model))
        self.activation = _get_activation(activation)
    def pos_embed(self, t, pos: Optional[Tensor], num_zeros=None):
        if num_zeros is None: 
            return t+pos
        else: 
            return t if pos is None else torch.cat(t[:num_zeros],(t[num_zeros:]+pos),dim=0)
    def forward(self, src, 
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                num_zeros = None):
        src2 = self.ln1(src)
        src = self.pos_embed(src2, pos=pos, num_zeros=num_zeros)
        # MHA block
        src2 = self.mha(src, src, src2,
                       attn_mask=src_mask,
                       key_padding_mask=src_key_padding_mask)[0]
        src2 = self.ln2(src2)
        src = src + self.dropout1(src2)
        src2 = self.ffn(src)
        src2 = self.ln3(src2)
        src = src + self.dropout2(src2)
        return src
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
