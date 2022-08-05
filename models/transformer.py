import copy
from numpy import np
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from tkinter import N
from turtle import forward
from typing import Optional
import torch
from torch import nn, Tensor
class TransformerEncoderLeft(nn.Module):
    def __init__(self, dim_model=512, nhead=8, num_enc_layers=6, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_verb_classes = 504
        self.verb_tokens = nn.Parameter(torch.zeros((1,dim_model)), requires_grad = True)
        # Encoder
        enc_layer = EncoderLeft_Layer(dim_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = EncoderLeft(enc_layer, num_enc_layers)
        
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
    def forward(self, img_ft, mask, pos_embed):
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
        # verb_ft = verb_ft.unsqueeze(0)
        
        return img_ft, verb_ft, verb_pred

class TransformerEncoderRight(nn.Module):
    def __init__(self, dim_model=512, nhead=8, num_enc_layers=6, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.dim_model = dim_model
        self.nhead = nhead
        self.num_verb_classes = 504
        # Encoder
        dec_layer = EncoderRight_Layer(dim_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = EncoderRight(dec_layer, num_enc_layers)
        
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
    def forward(self, img_ft, verb_ft, verb_pred,
                mask, role_token_embed,
                pos_embed, vidx_ridx,
                targets=None, inference=False):
        assert img_ft.shape[1] == self.dim_model
        assert img_ft.shape == pos_embed.shape
        device = img_ft.device
        max_num_roles = 6

        # restore tensor specs
        hw,bs,c = img_ft.shape
        # img_ft = img_ft.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        if not inference:
            selected_roles = targets["roles"]
        else:
            top1_verb = torch.topk(verb_pred, k=1, dim=1)[1].item()
            selected_roles = vidx_ridx[top1_verb]
        # Encoder second half
        num_roles = len(selected_roles)
        selected_role_tokens = role_token_embed[selected_roles].unsqueeze(1)
        selected_role_tokens = F.pad(selected_role_tokens, (0,0,0,0,0,max_num_roles-num_roles),mode="constant",value=0)
        src = torch.cat((verb_ft, selected_role_tokens), dim=0)
        src = self.decoder(src, img_ft, pos=pos_embed)
        verb_ft, noun_ft = src.split([1,6], dim=0)
        """
        verb_ft: (1,bs,dim)
        noun_ft: (m,bs,dim)      
        verb_pred: (1,504)
        """
        return verb_ft, noun_ft

class EncoderLeft(nn.Module):
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
class EncoderRight(nn.Module):
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
class EncoderLeft_Layer(nn.Module):
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
class EncoderRight_Layer(nn.Module):
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
class Decoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
    def forward(self, feature_list, role_embed_list):
        for layer in self.layers:
            feature_list = layer(feature_list, role_embed_list)
        return feature_list
class Decoder_Layer(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu", support_set_size=5):
        super().__init__()
        self.support_set_size = 5
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.norm4 = nn.LayerNorm(dim_model)
        self.ffn1 = nn.Sequential(nn.Linear(dim_model, dim_feedforward),
                                 _get_activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, dim_model))
        self.ffn2 = nn.Sequential(nn.Linear(dim_model, dim_feedforward),
                                 _get_activation(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, dim_model))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # unlike the Encoder part, the mha block here has batch_first=True
        self.mha = nn.MultiheadAttention(dim_model, nhead, dropout=dropout, batch_first=True)
        # Aggregation first concatenates tensors on the channel dimension, and then aggregates
        self.agg1 = nn.Sequential(nn.Linear(dim_model*self.support_set_size, dim_model), nn.Sigmoid())
        self.agg2 = nn.Sequential(nn.Linear(dim_model*self.support_set_size, dim_model), nn.Sigmoid())
    def forward(self, feature_list, role_embed_list):
        """
        features: list of intermediary representations
        role_embed: list of role embeddings
        verb_ft: (1,1,dim)
        noun_ft: (1,m,dim)
        role_embed: (1,m,dim)
        """
        # Part 1: conputing Semantic Relation Messages
        verb_msg_list = noun_msg_list = []
        for i in range(self.support_set_size + 1):
            m = feature_list[i].shape[1]-1
            verb_ft, noun_ft = feature_list[i].split([1,m],dim=1)
            if i==0: # for later use in Parts 2 & 3
                tgt_verb_ft, tgt_noun_ft = verb_ft, noun_ft
                tgt_num_roles = m
            role_embed = role_embed_list[i]
            src = torch.cat((verb_ft, noun_ft + role_embed), dim=1)
            src = self.mha(src, src, src)[0]
            verb_ft, noun_ft = src.split([1,m], dim=1)
            verb_msg_list.append(verb_ft)
            noun_msg_list.append(noun_ft)
        # Part 2: Refine noun with verb
        agg_verb_msg, final_noun_ft = torch.zeros(0)
        for i in range(1,self.support_set_size+1):
            agg_verb_msg = torch.cat((agg_verb_msg, verb_msg_list[i]), dim=2)
        agg_verb_msg = self.agg1(agg_verb_msg) # (1,1,K*dim) ---> (1,1,dim)

        for i in range(tgt_num_roles):
            verb_ft = agg_verb_msg # empty tensor for aggregation
            noun_ft = tgt_noun_ft[:, i, :]
            src = self.norm1(noun_ft + verb_ft)
            src = self.norm2(src + self.ffn1(src))
            final_noun_ft = torch.cat((final_noun_ft, src), dim=1)

        # Part 3: Refine verb with noun
        verb_ft = tgt_verb_ft
        noun_ft = torch.zeros(0)
        for i in range(noun_msg_list[0].shape[1]):
            noun_ft = torch.cat((noun_ft, noun_msg_list[0][:, i, :]), dim=2)
        noun_ft = self.agg2(noun_ft) # (1,1,K*dim) ---> (1,1,dim)
        src = self.norm3(verb_ft + noun_ft)
        final_verb_ft = self.norm4(src + self.ffn2(src))
        
        feature_list[0] = torch.cat((final_verb_ft, final_noun_ft), dim=1)
        return feature_list
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
def build_transformer_encoder_left(args):
    return TransformerEncoderLeft(args.hidden_dim, args.nheads,
                                  args.num_enc_layers, args.dim_feedforward, args.dropout)
def build_transformer_encoder_right(args):
    return TransformerEncoderRight(args.hidden_dim, args.nheads,
                                  args.num_enc_layers, args.dim_feedforward, args.dropout)