# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
CoFormer model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, accuracy_swig_bbox)
from .backbone import build_backbone
from .transformer import build_transformer_encoder, build_transformer_encoder


class GSRFormer_Encoder(nn.Module):
    """Encoder Part for GSRFormer"""
    def __init__(self, backbone, transformer, vidx_ridx):
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer encoder architecture. See transformer.py
            - vidx_ridx: verb index to role index (hash table)
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.vidx_ridx = vidx_ridx
        self.num_role_tokens = 190
        self.num_verb_tokens = 504

        # hidden dimension for tokens and image features
        hidden_dim = transformer.dim_model

        # token embeddings
        # self.role_token_embed = nn.Embedding(self.num_role_tokens, hidden_dim)
        # self.verb_token_embed = nn.Embedding(self.num_verb_tokens, hidden_dim)

        # 1x1 Conv
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1) # Same Specs as CoFormer
        
        # classifiers & predictors (for grounded noun prediction)
        # layer norms

    def forward(self, samples, targets=None, inference=False):
        """ 
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb' as the only key
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        batch_size = src.shape[0]
        batch_verb = []
        # model prediction
        for i in range(batch_size):
            if not inference:
                tgt = targets[i]
            else: 
                tgt = None
            outs = self.transformer(self.input_proj(src[i:i+1]), mask[i:i+1], 
                                    pos[-1][i:i+1], self.vidx_ridx, targets=tgt, inference=inference)
  
            # output features & predictions
            img_ft, verb_ft, verb_pred = outs[0], outs[1], outs[2]
            batch_verb.append(verb_pred)
        # outputs
        out = {}
        out['pred_verb'] = torch.cat(batch_verb, dim=0)
        return out


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing """
    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        Parameters:
                - smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SWiG_Criterion_Encoder(nn.Module):
    """ 
    Loss for the encoder-only training stage with SWiG dataset
    """
    def __init__(self, weight_dict, SWiG_json_train=None, SWiG_json_eval=None):
        """ 
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function_verb = LabelSmoothing(0.3)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval


    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation, and evaluation of CoFormer.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        # top-1 & top 5 verb acc and calculate verb loss 
        assert 'pred_verb' in outputs
        verb_pred_logits = outputs['pred_verb'].squeeze(1)
        batch_size = verb_pred_logits.shape[0]
        device = verb_pred_logits.device
        gt_verbs = torch.stack([t['verbs'] for t in targets])
        verb_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_loss = self.loss_function_verb(verb_pred_logits, gt_verbs)
        
        out = {}
        # losses 
        out['loss_vce'] = verb_loss
        # All metrics should be calculated per verb and averaged across verbs.
        ## In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        ### Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset. 
        ### We calculate metrics in this way for simple implementation in distributed data parallel setting. 

        # accuracies (for verb and noun)
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k]).mean()

        return out


def build_gsrformer_encoder(args):
    backbone = build_backbone(args)
    transformer = build_transformer_encoder(args)

    model = GSRFormer_Encoder(backbone,
                              transformer,
                              vidx_ridx=args.vidx_ridx)
    criterion = None

    if not args.inference:
        weight_dict = {'loss_vce': args.verb_loss_coef}
    
        if not args.test:
            criterion = SWiG_Criterion_Encoder(weight_dict=weight_dict, 
                                      SWiG_json_train=args.SWiG_json_train, 
                                      SWiG_json_eval=args.SWiG_json_dev)
        else:
            criterion = SWiG_Criterion_Encoder(weight_dict=weight_dict, 
                                      SWiG_json_train=args.SWiG_json_train, 
                                      SWiG_json_eval=args.SWiG_json_test)

    return model, criterion
