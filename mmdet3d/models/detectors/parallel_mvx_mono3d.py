import torch
import numpy as np
import cv2
import mmcv
from torch.nn import functional as F

from mmdet.models import DETECTORS
from mmdet.models.builder import build_head
from mmdet3d.models.builder import build_fusion_layer
from .mvx_faster_rcnn import DynamicMVXFasterRCNN
from mmcv.runner import force_fp32

IDX = 0


@DETECTORS.register_module()
class ParallelMVXMono3D(DynamicMVXFasterRCNN):

    def __init__(self, img_bbox_head, li_fusion_layer, **kwargs):
        super(ParallelMVXMono3D, self).__init__(**kwargs)

        train_cfg, test_cfg = kwargs['train_cfg'], kwargs['test_cfg']
        img_bbox_head.update(train_cfg=train_cfg.img)
        img_bbox_head.update(test_cfg=test_cfg.img)
        self.img_bbox_head = build_head(img_bbox_head)

        self.li_fusion_layer = build_fusion_layer(li_fusion_layer)

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

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size,
                                    img_feats, img_metas, points)
        pts_aux_feats = x.clone()
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x, pts_aux_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, pts_aux_feats = self.extract_pts_feat(
            points, img_feats, img_metas)
        if False:
            # TODO: implement IACTR
            img_feats = self.li_fusion_layer(img_feats)
        return (img_feats, pts_feats, pts_aux_feats)

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

    def forward_aux_train(self, pts_aux_feats, img_aux_feats,
                          gt_foreground_pts, gt_center_pts):
        losses = None
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
            cv2.imwrite('%06d_ori.png' % IDX, img)
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
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        #  self.input_visualize(img, gt_bboxes)
        #  self.input_visualize(img, gt_bboxes_cam)

        img_feats, pts_feats, pts_aux_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        losses_img = self.img_bbox_head.forward_train(
            img_feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_3d_cam,
            gt_labels_3d, centers2d, depths, attr_labels, gt_bboxes_ignore)

        losses.update(losses_img)

        losses_aux = self.forward_aux_train(pts_aux_feats, img_feats,
                                            gt_foreground_pts, gt_center_pts)
        #  losses.update(losses_aux)

        return losses
