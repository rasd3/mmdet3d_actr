# ------------------------------------------------------------------------
# Modified Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
ACTR module
"""
import argparse
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .position_encoding import (
    PositionEmbeddingSineSparse,
    PositionEmbeddingSine,
    PositionEmbeddingSineSparseDepth,
    PositionEmbeddingLearnedDepth,
)
from .actr_transformer import build_deformable_transformer
from .actr_utils import (
    accuracy,
    get_args_parser,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
    NestedTensor,
)


class ACTR(nn.Module):
    """This is the Deformable ACTR module that performs cross projection"""

    def __init__(
        self,
        transformer,
        num_channels,
        num_feature_levels,
        max_num_ne_voxel,
        pos_encode_method="image_coor",
    ):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            num_channels: [List] number of feature channels to bring from Depth Network Layer
            num_feature_levels: [int] number of feature level
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        num_backbone_outs = len(num_channels)
        self.num_backbone_outs = num_backbone_outs
        assert num_backbone_outs == num_feature_levels
        if num_feature_levels > 1:
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            ])

        prior_prob = 0.01
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # add
        self.max_num_ne_voxel = max_num_ne_voxel
        self.pos_encode_method = pos_encode_method
        assert self.pos_encode_method in ["image_coor", "depth", "depth_learn"]
        if self.pos_encode_method == "image_coor":
            self.q_position_embedding = PositionEmbeddingSineSparse(
                num_pos_feats=self.transformer.q_model // 2, normalize=True)
        elif self.pos_encode_method == "depth":
            self.q_position_embedding = PositionEmbeddingSineSparseDepth(
                num_pos_feats=self.transformer.q_model, normalize=True)
        elif self.pos_encode_method == "depth_learn":
            self.q_position_embedding = PositionEmbeddingLearnedDepth(
                num_pos_feats=self.transformer.q_model)

        self.v_position_embedding = PositionEmbeddingSine(
            num_pos_feats=hidden_dim // 2, normalize=True)

    def scatter_non_empty_voxel(self,
                                v_feat,
                                q_enh_feats,
                                q_idxs,
                                in_zeros=False):
        if in_zeros:
            s_feat = torch.zeros_like(v_feat)
        else:
            s_feat = v_feat

        for idx, (q_feat, q_idx) in enumerate(zip(q_enh_feats, q_idxs)):
            q_num = q_idx.shape[0]
            q_feat_t = q_feat.transpose(1, 0)
            s_feat[idx][:, q_idx[:, 0], q_idx[:, 1],
                        q_idx[:, 2]] = q_feat_t[:, :q_num]
        return s_feat

    def forward(
        self,
        v_feat,
        grid,
        i_feats,
        lidar_grid=None,
    ):
        """Parameters:
            v_feat: 3d coord sparse voxel features (B, C, X, Y, Z)
            grid: image coordinates of each v_features (B, X, Y, Z, 3)
            i_feats: image features (consist of multi-level)
            in_zeros: whether scatter to empty voxel or not

        It returns a dict with the following elements:
           - "srcs_enh": enhanced feature from camera coordinates
        """
        # get query feature & ref points
        q_feat_flattens = v_feat
        q_ref_coors = grid

        if self.pos_encode_method == "image_coor":
            q_pos = self.q_position_embedding(q_feat_flattens,
                                              q_ref_coors).transpose(1, 2)
        elif "depth" in self.pos_encode_method:
            q_depths = lidar_grid[..., 0].clone()
            q_pos = self.q_position_embedding(q_feat_flattens,
                                              q_depths).transpose(1, 2)

        # get image feature with reduced channel
        pos = []
        srcs = []
        masks = []
        for l, src in enumerate(i_feats):
            s_proj = self.input_proj[l](src)
            mask = torch.zeros(
                (s_proj.shape[0], s_proj.shape[2], s_proj.shape[3]),
                dtype=torch.bool,
                device=src.device,
            )
            pos_l = self.v_position_embedding(NestedTensor(s_proj, mask)).to(
                s_proj.dtype)
            pos.append(pos_l)
            srcs.append(s_proj)
            masks.append(mask)

        q_enh_feats = self.transformer(srcs, masks, pos, q_feat_flattens,
                                       q_pos, q_ref_coors)

        return q_enh_feats


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(model_cfg):
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script",
        parents=[get_args_parser()])
    args = parser.parse_args([])
    device = torch.device(args.device)

    # from yaml
    num_channels = model_cfg.num_channels
    args.query_num_feat = model_cfg.query_num_feat
    args.hidden_dim = model_cfg.query_num_feat
    args.enc_layers = model_cfg.num_enc_layers
    args.pos_encode_method = model_cfg.pos_encode_method
    args.max_num_ne_voxel = model_cfg.max_num_ne_voxel

    transformer = build_deformable_transformer(args)
    model = ACTR(
        transformer,
        num_feature_levels=args.num_feature_levels,
        num_channels=num_channels,
        max_num_ne_voxel=args.max_num_ne_voxel,
        pos_encode_method=args.pos_encode_method)

    return model
