import torch
import numpy as np
import cv2
import mmcv
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type
from mmdet3d.utils import kitti_vis
from mmdet3d.ops.spconv.structure import SparseConvTensor
from mmdet3d.models.fusion_layers.point_fusion import get_2d_coor
from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.builder import build_fusion_layer, build_loss
from .mvx_faster_rcnn import DynamicMVXFasterRCNN
from mmcv.runner import force_fp32
from mmcv.cnn import build_conv_layer, build_upsample_layer, build_norm_layer

IDX = 0


@DETECTORS.register_module()
class ParallelMVXMono3D(DynamicMVXFasterRCNN):

    def __init__(self,
                 img_bbox_head,
                 use_li_fusion_layer,
                 pts_li_fusion_layer=None,
                 pos_li_fusion_layer='kim',
                 loss_pts_w=1.,
                 loss_img_w=1.,
                 use_aux_pts_cls=True,
                 use_aux_pts_reg=True,
                 use_aux_img_cls=True,
                 use_aux_con_cls=True,
                 aux_pts_loss_cls=None,
                 aux_pts_loss_reg=None,
                 aux_img_loss_cls=None,
                 aux_con_loss_cls=None,
                 upsample_strides=[0.5, 1, 2, 4],
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 use_conv_for_no_stride=False,
                 **kwargs):
        super(ParallelMVXMono3D, self).__init__(**kwargs)

        train_cfg, test_cfg = kwargs['train_cfg'], kwargs['test_cfg']
        if train_cfg is not None:
            img_bbox_head.update(train_cfg=train_cfg.img)
        img_bbox_head.update(test_cfg=test_cfg.img)
        self.img_bbox_head = build_head(img_bbox_head)

        # build li fusion layer
        self.use_li_fusion_layer = use_li_fusion_layer
        self.pos_li_fusion_layer = pos_li_fusion_layer
        self.pts_li_fusion_layer = None
        if self.use_li_fusion_layer:
            self.pts_li_fusion_layer = build_fusion_layer(pts_li_fusion_layer)

        # assign pts loss & img loss weight
        self.loss_pts_w = torch.tensor(loss_pts_w).cuda()
        self.loss_img_w = torch.tensor(loss_img_w).cuda()

        # build aux layer
        self.num_classes = self.img_bbox_head.num_classes
        self.use_aux_pts_cls = use_aux_pts_cls
        self.use_aux_pts_reg = use_aux_pts_reg
        self.use_aux_img_cls = use_aux_img_cls
        self.use_aux_con_cls = use_aux_con_cls
        if use_aux_pts_cls:
            self.aux_pts_cls = torch.nn.Linear(
                64, self.num_classes, bias=False)
            self.aux_pts_loss_cls = build_loss(aux_pts_loss_cls)
        if use_aux_pts_reg:
            self.aux_pts_reg = torch.nn.Linear(64, 3, bias=False)
            self.aux_pts_loss_reg = build_loss(aux_pts_loss_reg)
        if use_aux_img_cls:
            in_channels = self.img_neck.out_channels
            out_channel = 64
            deblocks = []
            self.aux_img_loss_cls = build_loss(aux_img_loss_cls)
            for i in range(len(upsample_strides)):
                stride = upsample_strides[i]
                if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                    upsample_layer = build_upsample_layer(
                        upsample_cfg,
                        in_channels=in_channels,
                        out_channels=out_channel,
                        kernel_size=upsample_strides[i],
                        stride=upsample_strides[i])
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    upsample_layer = build_conv_layer(
                        conv_cfg,
                        in_channels=in_channels,
                        out_channels=out_channel,
                        kernel_size=stride,
                        stride=stride)
                deblock = nn.Sequential(
                    upsample_layer,
                    build_norm_layer(norm_cfg, out_channel)[1],
                    nn.ReLU(inplace=True))
                deblocks.append(deblock)
            self.aux_img_cls_agg = nn.ModuleList(deblocks)
            self.aux_img_cls = nn.Conv2d(out_channel * len(upsample_strides),
                                         self.num_classes, 1)
        if use_aux_con_cls:
            if self.num_classes == 1:
                self.aux_con_loss_cls = build_loss(aux_pts_loss_reg)
            else:
                self.aux_con_loss_cls = build_loss(aux_con_loss_cls)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points, img_feats, img_metas, train,
                         li_fusion_layer):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x, pts_lidar_feats, pts_aux_feats, img_feats = self.pts_middle_encoder(
            voxel_features,
            feature_coors,
            batch_size,
            img_feats,
            img_metas,
            points,
            ret_lidar_features=True,
            li_fusion_layer=li_fusion_layer,
            pos_li_fusion_layer=self.pos_li_fusion_layer)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x, pts_aux_feats, pts_lidar_feats, img_feats

    def extract_feat(self, points, img, img_metas, train):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, pts_aux_feats, pts_lidar_feats, img_feats = self.extract_pts_feat(
            points, img_feats, img_metas, train, self.pts_li_fusion_layer)

        if train:
            return img_feats, pts_feats, pts_aux_feats
        else:
            return img_feats, pts_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def forward_aux_train(self, pts_aux_feats, img_aux_feats, gt_bboxes_3d,
                          gt_labels_3d, img_mask, img_metas):
        """ Consist of pts, img and consistency loss
        pts - pointwise segmentation loss
            - center estimaion loss
        img - pixelwise segmentation loss
        con - differnece between pts seg with img seg
        """

        def get_pts_aux_target(points, gt_bboxes_3d, gt_labels_3d):
            # from AuxPointLabeler
            num_pts = points.shape[0]
            gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
            points_numpy = points.cpu().clone().numpy()
            foreground_masks = box_np_ops.points_in_rbbox(
                points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5))
            gt_foreground_pts = points.new_zeros(points.shape[0]).to(
                torch.long)
            gt_foreground_pts += self.num_classes
            pos_mask = torch.tensor(foreground_masks.sum(1) != 0)
            pos_inds = pos_mask.nonzero().squeeze()
            gt_foreground_pts_pos = gt_labels_3d[
                foreground_masks[pos_inds].argmax(1)]
            gt_foreground_pts[pos_inds] = gt_foreground_pts_pos

            gt_center_pts = np.zeros((num_pts, 3))
            for idx, gt_bbox in enumerate(gt_bboxes_3d_np):
                gt_center_pts[foreground_masks[:, idx]] = gt_bboxes_3d_np[
                    idx, :3] - points_numpy[foreground_masks[:, idx]][:, :3]
            gt_center_pts = torch.tensor(gt_center_pts)
            return gt_foreground_pts, gt_center_pts

        def sp2list(spt, batch_size):
            spt_list = []
            for idx in range(batch_size):
                mask = (spt.indices[:, 0] == idx)
                features = spt.features[mask]
                indices = spt.indices[mask]
                spatial_shape = spt.spatial_shape
                spt_list.append(
                    SparseConvTensor(features, indices, spatial_shape,
                                     batch_size))
            return spt_list

        def coor2pts(x):
            ratio = self.pts_middle_encoder.sparse_shape[1] / x.spatial_shape[1]
            pts = x.indices * torch.tensor(
                (self.pts_voxel_layer.voxel_size + [1])[::-1]).cuda() * ratio
            pts[:, 0] = pts[:, 0] / ratio
            pts[:, 1:] += torch.tensor(
                self.pts_voxel_layer.point_cloud_range[:3][::-1]).cuda()
            pts[:, 1:] = pts[:, [3, 2, 1]]
            return pts[:, 1:]

        def pts2img(spt, img_meta, img_shape):
            pts = coor2pts(spt)
            img_scale_factor = (
                pts.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                pts.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, 'LIDAR')
            proj_pts = get_2d_coor(
                img_meta=img_meta,
                points=pts,
                coord_type='LIDAR',
                proj_mat=pts.new_tensor(proj_mat),
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_meta['input_shape'][:2],
                img_shape=img_meta['input_shape'][:2])
            proj_pts = proj_pts[:, [1, 0]] * torch.tensor(img_shape).cuda()
            proj_pts = proj_pts.to(torch.long)
            proj_pts[:, 0] = torch.clamp(proj_pts[:, 0], 0, img_shape[0] - 1)
            proj_pts[:, 1] = torch.clamp(proj_pts[:, 1], 0, img_shape[1] - 1)
            return proj_pts

        losses = dict()
        batch_size = len(gt_bboxes_3d)

        # pts aux loss
        if self.use_aux_pts_cls * self.use_aux_pts_reg:
            pts_aux_feat_list = sp2list(pts_aux_feats, batch_size)
            aux_pts_losses_cls, aux_pts_losses_reg = torch.tensor(
                0.).cuda(), torch.tensor(0.).cuda()
            pred_cls_pts_list, pred_cls_img_list = [], []
            pos_pts_inds_list = []
            for b, (pts_aux_feats, gt_bbox_3d, gt_label_3d) in enumerate(
                    zip(pts_aux_feat_list, gt_bboxes_3d, gt_labels_3d)):
                num_gt_bbox = gt_bbox_3d.tensor.shape[0]
                points = coor2pts(pts_aux_feats)
                gt_cls_pts, gt_reg_pts = get_pts_aux_target(
                    points, gt_bbox_3d, gt_label_3d)
                pos_inds = gt_cls_pts.nonzero().squeeze()
                pos_pts_inds_list.append(pos_inds)
                pred_cls_pts = self.aux_pts_cls(pts_aux_feats.features)
                pred_cls_pts_list.append(pred_cls_pts)
                pred_reg_pts = self.aux_pts_reg(pts_aux_feats.features)
                aux_pts_loss_cls = self.aux_pts_loss_cls(
                    pred_cls_pts,
                    gt_cls_pts.to(torch.long).cuda(),
                    avg_factor=pos_inds.shape[0])

                aux_pts_loss_reg = self.aux_pts_loss_reg(
                    pred_reg_pts[pos_inds],
                    gt_reg_pts[pos_inds].cuda(),
                    avg_factor=pos_inds.shape[0])

                aux_pts_losses_cls += aux_pts_loss_cls
                aux_pts_losses_reg += aux_pts_loss_reg
            losses.update({
                'losses_aux_pts_cls': aux_pts_losses_cls,
                'losses_aux_pts_reg': aux_pts_losses_reg
            })

        # img aux loss
        if self.use_aux_img_cls:
            cat_list = []
            aux_img_losses_cls = torch.tensor(0.).cuda()
            for i in range(len(img_aux_feats)):
                cat_list.append(self.aux_img_cls_agg[i](img_aux_feats[i]))
            cat_feat = torch.cat(cat_list, dim=1)
            pred_cls_img = self.aux_img_cls(cat_feat)
            gt_cls_img = nn.functional.interpolate(
                img_mask, size=pred_cls_img.shape[2:])
            pred_cls_img_list = pred_cls_img.permute(0, 2, 3, 1).clone()
            pred_cls_img = pred_cls_img.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.num_classes)
            gt_cls_img = gt_cls_img.squeeze().reshape(batch_size, -1)
            for b in range(batch_size):
                aux_img_loss_cls = self.aux_img_loss_cls(
                    pred_cls_img[b], gt_cls_img[b].to(torch.long))
                aux_img_losses_cls += aux_img_loss_cls
            losses.update({'losses_aux_img_reg': aux_img_losses_cls})

        # consistency loss
        if self.use_aux_con_cls:
            aux_con_losses_cls = torch.tensor(0.).cuda()
            for b in range(batch_size):
                # cls
                p_pts_cls_pred = pred_cls_pts_list[b]
                i_img_cls_pred = pred_cls_img_list[b]
                img_shape = i_img_cls_pred.shape[0:2]
                proj_pts = pts2img(pts_aux_feat_list[b], img_metas[b],
                                   img_shape)

                fore_thres = 0.2
                p_img_cls_pred = i_img_cls_pred[proj_pts[:, 0], proj_pts[:, 1]]
                p_pts_mask = p_pts_cls_pred.max(1)[0] > fore_thres
                p_img_mask = p_img_cls_pred.max(1)[0] > fore_thres
                p_mask = torch.logical_or(p_pts_mask, p_img_mask)

                pts_cls_pred = p_pts_cls_pred[p_mask]
                img_cls_pred = p_img_cls_pred[p_mask]
                if p_mask.nonzero().shape[0] == 0:
                    continue
                if self.num_classes == 1:
                    aux_con_losses_cls += self.aux_con_loss_cls(
                        pts_cls_pred, img_cls_pred)
                else:
                    aux_con_losses_cls += self.aux_con_loss_cls(
                        pts_cls_pred, img_cls_pred)
                    aux_con_losses_cls += self.aux_con_loss_cls(
                        img_cls_pred, pts_cls_pred)
            losses.update({'aux_con_losses_cls': aux_con_losses_cls})

        return losses

    def input_visualize(self, imgs, gt_bboxes):
        img_norm_cfg = dict(
            mean=np.array([103.530, 116.280, 123.675]),
            std=np.array([1.0, 1.0, 1.0]),
            to_rgb=False)
        global IDX
        mean, std, to_rgb = img_norm_cfg['mean'], img_norm_cfg[
            'std'], img_norm_cfg['to_rgb']
        for idx, (img, bboxes) in enumerate(zip(imgs, gt_bboxes)):
            img = img.cpu().numpy() + mean.reshape(-1, 1, 1)
            img = np.transpose(img.astype(np.uint8), (1, 2, 0))
            bboxes = bboxes.detach().cpu().to(torch.int).numpy()
            for bbox in bboxes:
                img = cv2.rectangle(img.copy(), (bbox[0], bbox[1]),
                                    (bbox[2], bbox[3]), (0, 0, 255), 3)
            cv2.imwrite('./vis/%06d_ori.png' % IDX, img)
            IDX += 1

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_bboxes_3d_cam=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_cam=None,
                      gt_foreground_pts=None,
                      gt_center_pts=None,
                      img=None,
                      img_mask=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        #  self.input_visualize(img, gt_bboxes)
        #  self.input_visualize(img, gt_bboxes)

        img_feats, pts_feats, pts_aux_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, train=True)

        losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses_img = self.img_bbox_head.forward_train(
            img_feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_3d_cam,
            gt_labels_3d, centers2d, depths, attr_labels, gt_bboxes_ignore)
        losses_aux = self.forward_aux_train(pts_aux_feats, img_feats,
                                            gt_bboxes_3d, gt_labels_3d,
                                            img_mask, img_metas)

        losses = dict()
        for key in losses_pts:
            if type(losses_pts[key]) == list:
                losses_pts[key][0] *= self.loss_pts_w
            else:
                losses_pts[key] *= self.loss_pts_w
            losses.update({'pts_' + key: losses_pts[key]})
        for key in losses_img:
            losses.update({'img_' + key: losses_img[key] * self.loss_img_w})
        losses.update(losses_aux)

        return losses

    def simple_test_img(self, img_feats, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        img_metas[0]['scale_factor'] = torch.tensor([1.0, 1.0, 1.0, 1.0]).cuda()
        outs = self.img_bbox_head(img_feats)
        bbox_outputs = self.img_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if self.img_bbox_head.pred_bbox2d:
            from mmdet.core import bbox2result
            bbox2d_img = [
                bbox2result(bboxes2d, labels, self.img_bbox_head.num_classes)
                for bboxes, scores, labels, attrs, bboxes2d in bbox_outputs
            ]
            bbox_outputs = [bbox_outputs[0][:-1]]

        bbox_img = [
            bbox3d2result(bboxes, scores, labels, attrs)
            for bboxes, scores, labels, attrs in bbox_outputs
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.img_bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, train=False)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                if len(img_bbox) > 1:
                    for key in img_bbox.keys():
                        result_dict[key] = img_bbox[key]
                else:
                    result_dict['img_bbox'] = img_bbox
        return bbox_list
