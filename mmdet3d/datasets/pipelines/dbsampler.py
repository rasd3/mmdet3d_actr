# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import cv2
import numpy as np
import os
import pickle as pkl

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet.datasets import PIPELINES
from ..builder import OBJECTSAMPLERS


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False,
                 scene_list=None):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder
        self._scene_list = scene_list

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _sample_wimg(self, num, sample_idx):
        """Sample with img which has same calibration matrix with
        current sample's calibration matrix
        """
        ret = []
        while len(ret) < num:
            if self._idx == self._example_num:
                self._reset()
            sampled_dict = self._sampled_list[self._indices[self._idx]]
            if 'img_path' not in sampled_dict:
                self._idx += 1
                continue
            #  if self._scene_list[
            #  sampled_dict['image_idx']] != self._scene_list[sample_idx]:
            #  self._idx += 1
            #  continue
            ret.append(self._indices[self._idx])
            self._idx += 1
            if self._idx == self._example_num:
                self._reset()

        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num, with_img=False, sample_idx=None):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        if with_img:
            indices = self._sample_wimg(num, sample_idx)
        else:
            indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3]),
                 scene_path=None,
                 overlap_2d_thres=0.):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)
        # add
        self.with_img = scene_path is not None
        self.scene_list = None
        self.overlap_2d_thres = overlap_2d_thres
        if self.with_img:
            scene_list = pkl.load(open(scene_path, 'rb'))['sample']
            self.scene_list = scene_list

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(
                v, k, shuffle=True, scene_list=self.scene_list)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def img_update(self, img, gt_bboxes, gt_all_bboxes, gt_all_bboxes_3d,
                   sampled_gt_bboxes_2d, sampled_dict, img_dict):

        def img2mask(img):
            return img.sum(2) != 0

        def overlap_btw_mask(img1, img2, coor1, coor2):
            mask1, mask2 = img2mask(img1), img2mask(img2)
            lu = np.array([min(coor1[0], coor2[0]),
                           min(coor1[1], coor2[1])],
                          dtype=np.int)
            rd = np.array([
                max(coor1[0] + mask1.shape[0], coor2[0] + mask2.shape[0]),
                max(coor1[1] + mask1.shape[1], coor2[1] + mask2.shape[1])
            ],
                          dtype=np.int)
            mask_1 = np.zeros((rd[0] - lu[0], rd[1] - lu[1]), dtype=np.bool)
            mask_2 = np.zeros((rd[0] - lu[0], rd[1] - lu[1]), dtype=np.bool)
            mask_1[(coor1 - lu)[0]:(coor1 - lu)[0] + mask1.shape[0],
                   (coor1 - lu)[1]:(coor1 - lu)[1] + mask1.shape[1]] = mask1
            try:
                mask_2[(coor2 - lu)[0]:(coor2 - lu)[0] + mask2.shape[0],
                       (coor2 -
                        lu)[1]:(coor2 - lu)[1] + mask2.shape[1]] = mask2
            except:
                abcd = 1
            mask_btw = np.logical_and(mask_1, np.logical_not(mask_2))
            mask_b = mask_btw[(coor1 - lu)[0]:(coor1 - lu)[0] + mask1.shape[0],
                              (coor1 - lu)[1]:(coor1 - lu)[1] + mask1.shape[1]]
            return mask_b

        gt_num, gts_num = gt_bboxes.shape[0], gt_all_bboxes.shape[0]
        ov_iou = box_np_ops.iou_jit(gt_all_bboxes, gt_all_bboxes)
        diag = np.arange(gts_num)
        ov_iou[diag, diag] = 0.
        dist = np.sqrt(np.square(gt_all_bboxes_3d[:, :3]).sum(1))

        gts_img, gts_coor = [], []
        for i in range(gt_num):
            bbox, disc = gt_all_bboxes[i], img_dict['disc'][i]
            if img_dict['path'][i] == ' ':
                gts_img.append(np.array([0]))
            else:
                gts_img.append(cv2.imread(img_dict['path'][i]))
            gts_coor.append(
                np.array([bbox[1] + disc[1], bbox[0] + disc[0]], dtype=np.int))
        for i in range(gt_num, gts_num):
            samp = sampled_dict[i - gt_num]
            gts_img.append(cv2.imread(samp['img_path']))
            gts_coor.append(samp['img_bbox_coor'][:2][::-1])
        for i in range(gt_num, gts_num):
            samp = sampled_dict[i - gt_num]
            mask = img2mask(gts_img[i])
            for j in range(gts_num):
                if ov_iou[i, j] > 0.01 and dist[i] > dist[j]:
                    if gts_img[j].shape[0] == 1:
                        i_coor = np.array([
                            gts_coor[i][1], gts_coor[i][0],
                            gts_coor[i][1] + gts_img[i].shape[1],
                            gts_coor[i][0] + gts_img[i].shape[0]
                        ])
                        mask_btw = box_np_ops.mask_btw_2d_box(
                            i_coor, gt_all_bboxes[j].astype(np.int))
                        mask = np.logical_and(mask_btw, mask)
                    else:
                        mask_btw = overlap_btw_mask(gts_img[i], gts_img[j],
                                                    gts_coor[i], gts_coor[j])
                        mask = np.logical_and(mask_btw, mask)
            cut_x = min(img.shape[0] - gts_coor[i][0] - gts_img[i].shape[0], 0)
            cut_y = min(img.shape[1] - gts_coor[i][1] - gts_img[i].shape[1], 0)
            if mask.shape[0] + cut_x <= 0 or mask.shape[1] + cut_y <= 0:
                continue
            mask = mask[:mask.shape[0] + cut_x, :mask.shape[1] + cut_y]
            if mask.sum() == 0:
                continue
            img[gts_coor[i][0]:gts_coor[i][0] + gts_img[i].shape[0] + cut_x,
                gts_coor[i][1]:gts_coor[i][1] + gts_img[i].shape[1] +
                cut_y][mask] = gts_img[i][:gts_img[i].shape[0] +
                                          cut_x, :gts_img[i].shape[1] +
                                          cut_y][mask]

        return img

    def sample_all(
        self,
        gt_bboxes,
        gt_labels,
        gt_bboxes_2d=None,
        gt_bboxes_3d_cam=None,
        gt_bboxes_depths=None,
        gt_bboxes_centers2d=None,
        sample_idx=None,
        img=None,
        img_dict=None,
    ):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.
            gt_bboxes_2d (np.ndarray): Ground truth labels of 2D boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []

        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        sampled_gt_bboxes_2d = []
        avoid_coll_boxes_2d = gt_bboxes_2d
        sampled_gt_bboxes_cam = []
        avoid_coll_boxes_cam = gt_bboxes_3d_cam.tensor.numpy(
        ) if gt_bboxes_3d_cam is not None else None
        sampled_centers2d = []
        avoid_coll_centers2d = gt_bboxes_centers2d
        sampled_depths = []
        avoid_coll_depths = gt_bboxes_depths

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes,
                                                   avoid_coll_boxes_2d,
                                                   sample_idx)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    if gt_bboxes_3d_cam is not None:
                        if len(sampled_cls) == 1:
                            sampled_gt_box_cam = sampled_cls[0][
                                'box3d_cam'].tensor.numpy()[np.newaxis, ...]
                        else:
                            sampled_gt_box_cam = np.stack([
                                s['box3d_cam'].tensor.numpy()
                                for s in sampled_cls
                            ],
                                                          axis=0)
                        sampled_gt_bboxes_cam += [sampled_gt_box_cam]
                        avoid_coll_boxes_cam = np.concatenate(
                            [avoid_coll_boxes_cam, sampled_gt_box_cam[:, 0]],
                            axis=0)

                    if gt_bboxes_depths is not None:
                        if len(sampled_cls) == 1:
                            sampled_center2d = sampled_cls[0]['centers2d'][
                                np.newaxis, ...]
                            sampled_depth = sampled_cls[0]['depths'][
                                np.newaxis, ...]
                        else:
                            sampled_center2d = np.stack(
                                [s['centers2d'] for s in sampled_cls], axis=0)
                            sampled_depth = np.stack(
                                [s['depths'] for s in sampled_cls], axis=0)

                        sampled_depths += [sampled_depth]
                        avoid_coll_depths = np.concatenate(
                            [avoid_coll_depths, sampled_depth], axis=0)
                        sampled_centers2d += [sampled_center2d]
                        avoid_coll_centers2d = np.concatenate(
                            [avoid_coll_centers2d, sampled_center2d], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

                    if self.with_img:
                        if len(sampled_cls) == 1:
                            sampled_gt_box_2d = sampled_cls[0]['bbox'][
                                np.newaxis, ...]
                        else:
                            sampled_gt_box_2d = np.stack(
                                [s['bbox'] for s in sampled_cls], axis=0)
                        sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                        avoid_coll_boxes_2d = np.concatenate(
                            [avoid_coll_boxes_2d, sampled_gt_box_2d], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            s_points_list = []
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)
            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

            if self.with_img:
                sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d)
                img = self.img_update(img, gt_bboxes, avoid_coll_boxes_2d,
                                      avoid_coll_boxes, sampled_gt_bboxes_2d,
                                      sampled, img_dict)
                ret.update({'img': img, 'gt_bboxes_2d': sampled_gt_bboxes_2d})
            if gt_bboxes_3d_cam is not None:
                sampled_gt_bboxes_cam = np.concatenate(sampled_gt_bboxes_cam)
                ret.update({
                    'gt_bboxes_3d_cam': sampled_gt_bboxes_cam,
                })
            if gt_bboxes_depths is not None:
                sampled_centers2d = np.concatenate(sampled_centers2d)
                sampled_depths = np.concatenate(sampled_depths)
                ret.update({
                    'centers2d': sampled_centers2d,
                    'depths': sampled_depths
                })

        return ret

    def sample_class_v2(self,
                        name,
                        num,
                        gt_bboxes,
                        gt_bboxes_2d=None,
                        sample_idx=None):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num, self.with_img,
                                                 sample_idx)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        valid_check = [False for _ in range(num_gt + num_sampled)]
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
                valid_check[i] = False
            else:
                valid_check[i] = True

        if self.with_img:
            cv_boxes = np.stack([i['bbox'] for i in sampled], axis=0)
            total_cv = np.concatenate([gt_bboxes_2d, cv_boxes], axis=0)
            overlaps_iou = box_np_ops.overlap_jit(total_cv)
            #  overlaps_iou = box_np_ops.iou_jit(total_cv, total_cv)
            overlaps_iou[diag, diag] = 0.
            for i in range(num_gt, num_gt + num_sampled):
                if overlaps_iou[i].max() > self.overlap_2d_thres:
                    overlaps_iou[i] = 0.
                    overlaps_iou[:, i] = 0.
                    valid_check[i] = False

        for i in range(num_gt, num_gt + num_sampled):
            if valid_check[i]:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples


def legacy_code_for_imgt_collision_test():
    """
    s_dict['bbox'][2] = min(s_dict['bbox'][2], img.shape[1])
    s_dict['bbox'][3] = min(s_dict['bbox'][3], img.shape[0])
    if s_dict['bbox'][0] > img.shape[1] or s_dict['bbox'][1] > img.shape[0]:
        continue
    s_img = cv2.imread(self.data_root + s_dict['img_path'])
    mask = np.zeros(
        (int(s_dict['bbox'][3]) - int(s_dict['bbox'][1]),
         int(s_dict['bbox'][2]) - int(s_dict['bbox'][0])),
        dtype=np.bool)
    b_img = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask1 = np.logical_not(
        (s_img == np.array([0, 0, 0])).sum(2) == 3)
    sx = max(s_dict['img_bbox_coor'][0] - int(s_dict['bbox'][0]), 0)
    sy = max(s_dict['img_bbox_coor'][1] - int(s_dict['bbox'][1]), 0)
    i_shape = mask[sy:sy + s_img.shape[0],
                   sx:sx + s_img.shape[1]].shape
    if s_img.shape[:2] != i_shape:
        s_img = s_img[:i_shape[0], :i_shape[1]]
        mask1 = mask1[:i_shape[0], :i_shape[1]]
    mask[sy:sy + s_img.shape[0],
         sx:sx + s_img.shape[1]] = mask1

    b_img[sy:sy + s_img.shape[0],
          sx:sx + s_img.shape[1]] = s_img
    for j in range(gts_num):
        if ov_iou[i, j] > 0.01 and dist[i] > dist[j]:
            mask_btw = box_np_ops.mask_btw_2d_box(
                s_dict['bbox'].astype(np.int),
                avoid_coll_boxes_2d[j].astype(np.int))
            mask = np.logical_and(mask_btw, mask)
    s_box = s_dict['bbox'].astype(np.int)
    dst = img[s_box[1]:s_box[3], s_box[0]:s_box[2]]
    dst[mask] = b_img[mask]
    img[s_box[1]:s_box[3], s_box[0]:s_box[2]] = dst
    """
    pass
