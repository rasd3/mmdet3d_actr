# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import cv2
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from mmdet3d.models.model_utils.actr import build as build_actr
from mmdet3d.models.model_utils.pointformer import LocalTransformer, LocalGlobalTransformer
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation

IDX = 0


class BasicGate(nn.Module):
    # mod code from 3D-CVF
    def __init__(self, in_channel, convf='Conv1d'):
        super(BasicGate, self).__init__()
        self.in_channel = in_channel
        if convf == 'Conv1d':
            conv_func = nn.Conv1d
        if convf == 'Conv2d':
            conv_func = nn.Conv2d
        self.gating_conv = conv_func(
            self.in_channel,
            1,
            kernel_size=1,
            stride=1,
        )

    def forward(self, src, trg):
        g_map = torch.sigmoid(self.gating_conv(src))
        return trg * g_map


def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()


def get_2d_coor(img_meta, points, proj_mat, coord_type, img_scale_factor,
                img_crop_offset, img_flip, img_pad_shape, img_shape):
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h
    coor_x = coor_x / w
    grid = torch.cat([coor_x, coor_y], dim=1)  # Nx2 -> 1x1xNx2
    grid = torch.clamp(grid, min=0.0, max=1.0)
    return grid


@FUSION_LAYERS.register_module()
class PointFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts


@FUSION_LAYERS.register_module()
class ACTR(BaseModule):

    def __init__(self,
                 actr_cfg,
                 init_cfg=None,
                 coord_type='LIDAR',
                 activate_out=False,
                 model_name='ACTR',
                 lt_cfg=None):
        super(ACTR, self).__init__(init_cfg=init_cfg)
        self.fusion_method = actr_cfg['fusion_method']
        self.actr = build_actr(actr_cfg, model_name=model_name, lt_cfg=lt_cfg)
        self.coord_type = coord_type
        self.activate_out = activate_out
        if self.fusion_method == 'gating_v1':
            n_channel = actr_cfg['query_num_feat']
            self.trg_gating = BasicGate(n_channel)
            self.trg_channel_reduce = nn.Conv1d(
                n_channel * 2, n_channel, kernel_size=1, stride=1)
        if False:
            self.lt = LocalTransformer(2048, 2.0, 32, 64, 64, 4, 2,
                                       dict(type='BN2d'), 1, 0.0)
            self.lgt = LocalGlobalTransformer(64, 64, 4, 2, dict(type='BN2d'), 1,
                                              2048, 0.2, True)

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.
        Returns:
            torch.Tensor: Fused features of each point.
        """
        if False:
            np = pts[0].shape[0]
            xyz, features = pts[0].unsqueeze(0), pts_feats[:np].unsqueeze(
                0).transpose(2, 1)
            breakpoint()
            enh_feats = self.lt(xyz, features)

        batch_size = len(pts)
        img_feats = img_feats[:self.actr.num_backbone_outs]
        num_points = [i.shape[0] for i in pts]
        pts_feats_b = torch.zeros(
            (batch_size, self.actr.max_num_ne_voxel, pts_feats.shape[1]),
            device=pts_feats.device)
        coor_2d_b = torch.zeros((batch_size, self.actr.max_num_ne_voxel, 2),
                                device=pts_feats.device)
        pts_b = torch.zeros((batch_size, self.actr.max_num_ne_voxel, 3),
                            device=pts_feats.device)

        for b in range(batch_size):
            img_meta = img_metas[b]
            img_scale_factor = (
                pts[b].new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                pts[b].new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            coor_2d = get_2d_coor(
                img_meta=img_meta,
                points=pts[b][:, :3],
                coord_type=self.coord_type,
                proj_mat=pts[b].new_tensor(proj_mat),
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_meta['input_shape'][:2],
                img_shape=img_meta['img_shape'][:2])
            pts_b[b, :pts[b].shape[0]] = pts[b][:, :3]
            coor_2d_b[b, :pts[b].shape[0]] = coor_2d
            pts_feats_b[b, :pts[b].shape[0]] = pts_feats[b]

        enh_feat = self.actr(
            v_feat=pts_feats_b,
            grid=coor_2d_b,
            i_feats=img_feats,
            lidar_grid=pts_b,
        )
        enh_feat_cat = torch.cat(
            [f[:np] for f, np in zip(enh_feat, num_points)])

        if self.fusion_method == 'replace':
            fuse_out = enh_feat_cat
        elif self.fusion_method == 'concat':
            fuse_out = torch.cat((pts_feats, enh_feat_cat), dim=1)
        elif self.fusion_method == 'sum':
            fuse_out = pts_feats + enh_feat_cat
        elif self.fusion_method == 'gating_v1':
            pts_feats_u = pts_feats.unsqueeze(0).permute(0, 2, 1)
            enh_feat_cat_u = enh_feat_cat.unsqueeze(0).permute(0, 2, 1)
            gated_fuse_feat_u = self.trg_gating(pts_feats_u, enh_feat_cat_u)
            fuse_out = pts_feats_u + enh_feat_cat_u
            fuse_out = fuse_out.squeeze().permute(1, 0)
            #  fuse_out = torch.cat((pts_feats_u, gated_fuse_feat_u), dim=1)
            #  fuse_out = self.trg_channel_reduce(fuse_out)
            #  fuse_out = fuse_out.squeeze().permute(1, 0)
        else:
            NotImplementedError('Invalid ACTR fusion method')

        if self.activate_out:
            fuse_out = F.relu(fuse_out)

        return fuse_out


@FUSION_LAYERS.register_module()
class IACTR(BaseModule):

    def __init__(
        self,
        actr_cfg,
        init_cfg=None,
        coord_type='LIDAR',
        activate_out=False,
        voxel_size=None,
        sparse_shape=None,
        point_cloud_range=None,
        ret_pts_img=False,
        model_name='IACTR',
        make_depth_img=False,
    ):
        super(IACTR, self).__init__(init_cfg=init_cfg)
        self.fusion_method = actr_cfg['fusion_method']
        self.iactr = build_actr(actr_cfg, model_name=model_name)
        self.model_name = model_name
        self.make_depth_img = make_depth_img
        self.coord_type = coord_type
        self.activate_out = activate_out
        self.voxel_size = voxel_size
        self.sparse_shape = sparse_shape
        self.point_cloud_range = point_cloud_range
        self.ret_pts_img = ret_pts_img
        if self.fusion_method == 'gating_v1':
            n_channel = actr_cfg['query_num_feat']
            self.trg_gating = BasicGate(n_channel, convf='Conv2d')
            self.trg_channel_reduce = nn.ModuleList()
            for i in range(4):
                self.trg_channel_reduce.append(
                    nn.Conv2d(
                        n_channel * 2, n_channel, kernel_size=1, stride=1))

    def visualize(self, pts_feat):
        global IDX
        pts_feat = pts_feat.detach().cpu().max(2)[0].numpy()
        pts_feat = (pts_feat * 255.).astype(np.uint8)
        cv2.imwrite('lidar2img_%d.png' % IDX, pts_feat)
        IDX += 1

    def coor2pts(self, x):
        ratio = self.sparse_shape[1] / x.spatial_shape[1]
        pts = x.indices * torch.tensor(
            (self.voxel_size + [1])[::-1]).cuda() * ratio
        pts[:, 0] = pts[:, 0] / ratio
        pts[:, 1:] += torch.tensor(self.point_cloud_range[:3][::-1]).cuda()
        pts[:, 1:] = pts[:, [3, 2, 1]]
        return pts[:, 1:]

    def pts2img(self, coor, pts_feat, shape, pts, ret_depth=False):
        coor = coor[:, [1, 0]]
        i_shape = torch.cat(
            [shape + 1,
             torch.tensor([pts_feat.features.shape[1]]).cuda()])
        i_pts_feat = torch.zeros(tuple(i_shape), device=coor.device)
        i_coor = (coor * shape).to(torch.long)
        i_pts_feat[i_coor[:, 0], i_coor[:, 1]] = pts_feat.features
        i_pts_feat = i_pts_feat[:-1, :-1].permute(2, 0, 1)
        #  self.viualize(i_pts_feat)
        if ret_depth:
            i_shape[2] = 3
            i_depth_feat = torch.zeros(tuple(i_shape), device=coor.device)
            i_depth_feat[i_coor[:, 0], i_coor[:, 1]] = pts
            i_depth_feat = i_depth_feat[:-1, :-1].permute(2, 0, 1)
            return i_pts_feat, i_depth_feat

        return i_pts_feat, None

    def forward(self, img_feats, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.
        Returns:
            torch.Tensor: Fused features of each point.
        """
        batch_size = len(img_metas)
        scale_size = len(pts_feats)
        device = img_feats[0].device
        img_feats = img_feats[:self.iactr.num_backbone_outs]
        img_shapes = [
            torch.tensor(f.shape[2:], device=device) for f in img_feats
        ]

        pts_img_list = []
        pts_depth_list = []
        for s in range(scale_size):
            batch_img_list = []
            batch_depth_list = []
            for b in range(batch_size):
                img_meta = img_metas[b]
                img_scale_factor = (
                    img_feats[b].new_tensor(img_meta['scale_factor'][:2])
                    if 'scale_factor' in img_meta.keys() else 1)
                img_flip = img_meta['flip'] if 'flip' in img_meta.keys(
                ) else False
                img_crop_offset = (
                    img_feats[b].new_tensor(img_meta['img_crop_offset'])
                    if 'img_crop_offset' in img_meta.keys() else 0)
                proj_mat = get_proj_mat_by_coord_type(img_meta,
                                                      self.coord_type)
                pts = self.coor2pts(pts_feats[s])
                coor_2d = get_2d_coor(
                    img_meta=img_meta,
                    points=pts,
                    coord_type=self.coord_type,
                    proj_mat=pts[b].new_tensor(proj_mat),
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img_meta['input_shape'][:2],
                    img_shape=img_meta['img_shape'][:2])
                pt_img, pt_depth_img = self.pts2img(coor_2d, pts_feats[s],
                                                    img_shapes[s], pts,
                                                    self.make_depth_img)
                batch_img_list.append(pt_img.unsqueeze(0))
                if pt_depth_img is not None:
                    batch_depth_list.append(pt_depth_img.unsqueeze(0))
            pts_img_list.append(torch.cat(batch_img_list))
            if len(batch_depth_list):
                pts_depth_list.append(torch.cat(batch_depth_list))

        if self.model_name == 'IACTRv3':
            enh_feat = self.iactr(
                i_feats=img_feats,
                p_feats=pts_img_list,
                p_depths=pts_depth_list,
                ret_pts_img=self.ret_pts_img)
        else:
            enh_feat = self.iactr(
                i_feats=img_feats,
                p_feats=pts_img_list,
                ret_pts_img=self.ret_pts_img)
        if self.fusion_method == 'replace':
            pass
        elif self.fusion_method == 'sum':
            for s in range(scale_size):
                enh_feat[s] = img_feats[s] + enh_feat[s]
        elif self.fusion_method == 'sum_coor':
            for s in range(scale_size):
                enh_feat[s] = img_feats[s] + enh_feat[s]
                enh_feat[s] = torch.cat((enh_feat[s], pts_depth_list[s]),
                                        dim=1)
        elif self.fusion_method == 'gating_v1':
            for s in range(scale_size):
                gated_fuse_feat = self.trg_gating(img_feats[s], enh_feat[s])
                #  fuse_out = torch.cat((img_feats[s], gated_fuse_feat), dim=1)
                #  fuse_out = self.trg_channel_reduce[s](fuse_out)
                enh_feat[s] = img_feats[s] + gated_fuse_feat
        elif self.fusion_method == 'cat':
            for s in range(scale_size):
                enh_feat[s] = torch.cat((img_feats[s], enh_feat[s]), dim=1)

        return enh_feat
