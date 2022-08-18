# ----------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

from .gsrformer import build_gsrformer_encoder
from .gsrformer import build_gsrformer_decoder

def build_encoder_model(args):
    return build_gsrformer_encoder(args)
def build_decoder_model(all_features, all_role_tokens, args):
    return build_gsrformer_decoder(all_features, all_role_tokens, args)