from mmdet.models import DETECTORS
from mmdet.models.builder import build_head, build_fusion_layer
from .mvx_faster_rcnn import DynamicMVXFasterRCNN


@DETECTORS.register_module()
class ParallelMVXMono3D(DynamicMVXFasterRCNN):

    def __init__(self, img_bbox_head, li_fusion_layer, **kwargs):
        super(ParallelMVXMono3D, self).__init__(**kwargs)

        img_bbox_head.update(train_cfg=train_cfg.img)
        img_bbox_head.update(test_cfg=test_cfg.img)
        self.img_bbox_head = build_head(img_bbox_head)

        self.li_fusion_layer = builder.build_fusion_layer(li_fusion_layer)

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        if type(self.pts_voxel_encoder).__name__ == 'HardSimpleVFE':
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        else:
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                    img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size,
                                    img_feats, img_metas, pts)
        pts_aux_feat = x.clone()
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x, pts_aux_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        img_feats = self.li_fusion_layer(img_feats)
        pts_feats, pts_aux_feats = self.extract_pts_feat(
            points, img_feats, img_metas)
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

    def forward_aux_train(self, pts_aux_feats, img_aux_feats):
        losses = None
        return losses


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        img_feats, pts_feats, pts_aux_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        losses_img = self.img_bbox_head.forward_train(
            img_feats, img_metas, gt_bboxes, gt_labels, gt_bboxes_3d,
            gt_labels_3d, centers2d, depths, attr_labels, gt_bboxes_ignore)

        losses.update(losses_img)

        losses_aux = self.forward_aux_train(pts_aux_feats, img_feats)
        losses.update(losses_aux)

        return losses
