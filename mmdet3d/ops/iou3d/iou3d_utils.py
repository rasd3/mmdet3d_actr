import torch
import numpy as np

from . import iou3d_cuda
from mmdet3d.core.bbox import box_np_ops

try:
    from mmdet3d.ops.nms.nms import (
        non_max_suppression_cpu,
        rotate_non_max_suppression_cpu,
        IOU_weighted_rotate_non_max_suppression_cpu,
    )
except:
    current_dir = Path(__file__).resolve().parents[0]
    load_pb11(
        ["./nms_kernel.cu.cc", "./nms.cc"],
        current_dir / "nms.so",
        current_dir,
        cuda=True,
    )
    from mmdet3d.ops.nms.nms import (
        non_max_suppression_cpu,
        rotate_non_max_suppression_cpu,
        IOU_weighted_rotate_non_max_suppression_cpu,
    )


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()
    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh,
                                        boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()


def boxes3d_to_bev_torch(boxes3d, box_mode='wlh', rect=False):
    """
    Input(torch):
        boxes3d: (N, 7) [x, y, z, h, w, l, ry]
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry/rz], left-bottom: (x1, y1), right-top: (x2, y2), ry/rz: clockwise rotation angle
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))
    if boxes3d.shape[-1] == 5:
        w_index, l_index = box_mode.index('w') + 2, box_mode.index('l') + 2
    elif boxes3d.shape[-1] == 7:
        w_index, l_index = box_mode.index('w') + 3, box_mode.index('l') + 3
    else:
        raise NotImplementedError

    half_w, half_l = boxes3d[:, w_index] / 2., boxes3d[:, l_index] / 2.
    if rect:
        cu, cv = boxes3d[:, 0], boxes3d[:, 2]  # cam coord: x, z
        boxes_bev[:,
                  0], boxes_bev[:,
                                1] = cu - half_l, cv - half_w  # left-bottom in cam coord
        boxes_bev[:,
                  2], boxes_bev[:,
                                3] = cu + half_l, cv + half_w  # right-top in cam coord
    else:
        cu, cv = boxes3d[:, 0], boxes3d[:, 1]  # velo coord: x, y
        boxes_bev[:,
                  0], boxes_bev[:,
                                1] = cu - half_w, cv - half_l  # left-bottom in velo coord
        boxes_bev[:,
                  2], boxes_bev[:,
                                3] = cu + half_w, cv + half_l  # right-top in cam coord
    # rz in velo should have the same effect as ry of cam coord in 2D. Points in box will clockwisely rotate with rz/ry angle.
    boxes_bev[:, 4] = boxes3d[:, -1]
    return boxes_bev


def boxes_aligned_iou3d_gpu(boxes_a,
                            boxes_b,
                            box_mode='wlh',
                            rect=False,
                            need_bev=False):
    """
    Input (torch):
        boxes_a: (N, 7) [x, y, z, w, l, h, ry], torch tensor with type float32.
        boxes_b: (N, 7) [x, y, z, w, l, h, ry], torch tensor with type float32.
        rect: True/False means boxes in camera/velodyne coord system.
        Notice: (x, y, z) are real center.
    Output:
        iou_3d: (N)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index(
        'l') + 3, box_mode.index('h') + 3
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a, box_mode, rect)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b, box_mode, rect)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size(
        (boxes_a.shape[0], 1))).zero_()  # (N, 1)
    iou3d_cuda.boxes_aligned_overlap_bev_gpu(boxes_a_bev.contiguous(),
                                             boxes_b_bev.contiguous(),
                                             overlaps_bev)

    # bev iou
    area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(-1, 1)  # (N, 1)
    iou_bev = overlaps_bev / torch.clamp(
        area_a + area_b - overlaps_bev, min=1e-7)  # [N, 1]

    # height overlap
    if rect:
        raise NotImplementedError
        # boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, h_index]).view(-1, 1)  # y - h
        # boxes_a_height_max = boxes_a[:, 1].view(-1, 1)                    # y
        # boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, h_index]).view(1, -1)
        # boxes_b_height_max = boxes_b[:, 1].view(1, -1)
    else:
        half_h_a = boxes_a[:, h_index] / 2.0
        half_h_b = boxes_b[:, h_index] / 2.0
        if False:
            # todo: notice if (x, y, z) is the real center
            boxes_a_height_min = (boxes_a[:, 2] - half_h_a).view(
                -1, 1)  # z - h/2, (N, 1)
            boxes_a_height_max = (boxes_a[:, 2] + half_h_a).view(
                -1, 1)  # z + h/2, (N, 1)
            boxes_b_height_min = (boxes_b[:, 2] - half_h_b).view(-1, 1)
            boxes_b_height_max = (boxes_b[:, 2] + half_h_b).view(-1, 1)
        else:
            # (x, y, z) is bottom center
            boxes_a_height_min = (boxes_a[:, 2]).view(-1, 1)
            boxes_a_height_max = (boxes_a[:, 2] + half_h_a * 2).view(-1, 1)
            boxes_b_height_min = (boxes_b[:, 2]).view(-1, 1)
            boxes_b_height_max = (boxes_b[:, 2] + half_h_b * 2).view(-1, 1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)  # (N, 1)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)  # (N, 1)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)  # (N, 1)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1,
                                                                 1)  # (N, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1,
                                                                 1)  # (N, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    if need_bev:
        return iou3d, iou_bev

    return iou3d


def boxes_iou3d_gpu(boxes_a,
                    boxes_b,
                    box_mode='wlh',
                    rect=False,
                    need_bev=False):
    """
    #todo: Zheng Wu 20191024: If h, w, l, ry take the same metrics, I think it doesn't matter for boxes whether in velo or rect coord
    Input (torch):
        boxes_a: (N, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        boxes_b: (M, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        iou_3d: (N, M)
    """
    w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index(
        'l') + 3, box_mode.index('h') + 3
    boxes_a_bev = boxes3d_to_bev_torch(boxes_a, box_mode, rect)
    boxes_b_bev = boxes3d_to_bev_torch(boxes_b, box_mode, rect)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(),
                                     boxes_b_bev.contiguous(), overlaps_bev)

    # bev iou
    area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(
        1, -1)  # (1, M)  -> broadcast (N, M)
    iou_bev = overlaps_bev / torch.clamp(
        area_a + area_b - overlaps_bev, min=1e-7)

    # height overlap
    if rect:
        boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, h_index]).view(
            -1, 1)  # y - h
        boxes_a_height_max = boxes_a[:, 1].view(-1, 1)  # y
        boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, h_index]).view(1, -1)
        boxes_b_height_max = boxes_b[:, 1].view(1, -1)
    else:
        # todo: notice if (x, y, z) is the real center
        half_h_a = boxes_a[:, h_index] / 2.0
        half_h_b = boxes_b[:, h_index] / 2.0
        boxes_a_height_min = (boxes_a[:, 2] - half_h_a).view(
            -1, 1)  # z - h/2, (N, 1)
        boxes_a_height_max = (boxes_a[:, 2] + half_h_a).view(
            -1, 1)  # z + h/2, (N, 1)
        boxes_b_height_min = (boxes_b[:, 2] - half_h_b).view(-1, 1)
        boxes_b_height_max = (boxes_b[:, 2] + half_h_b).view(-1, 1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)  # (N, 1)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)  # (1, M)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)  # (N, M)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h  # broadcast: (N, M)

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1,
                                                                 1)  # (N, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(
        1, -1)  # (1, M)  -> broadcast (N, M)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    if need_bev:
        return iou3d, iou_bev
    return iou3d


def rotate_weighted_nms(
    box_preds,
    rbboxes,
    dir_labels,
    labels_preds,
    scores,
    iou_preds,
    anchors,
    pre_max_size=None,
    post_max_size=None,
    iou_threshold=0.5,
):

    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        rbboxes = rbboxes[indices]
        iou_preds = iou_preds[indices]
        dir_labels = dir_labels[indices]
        labels_preds = labels_preds[indices]
        box_preds = box_preds[indices]
        anchors = anchors[indices]

    dets = torch.cat([rbboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    iou_preds_np = iou_preds.data.cpu().numpy()
    dir_labels_np = dir_labels.cpu().numpy()
    labels_preds_np = labels_preds.cpu().numpy()
    box_preds_np = box_preds.cpu().numpy()
    scores_np = scores.cpu().numpy()
    anchors_np = anchors.cpu().numpy()

    if len(dets_np) == 0:
        box_ret_np, dir_ret_list, labels_ret_list, scores_ret_list = [
            np.array([], dtype=np.int64)
        ] * 4
    else:
        nms_result = rotate_weighted_nms_cc(
            box_preds_np,
            dets_np,
            iou_threshold,
            iou_preds_np,
            labels_preds_np,
            dir_labels_np,
            anchors_np,
        )
        box_ret_np = np.array(nms_result[0])
        scores_ret_np = np.array(nms_result[1])
        labels_ret_np = np.array(nms_result[2])
        dir_ret_np = np.array(nms_result[3])
    return torch.from_numpy(box_ret_np).cuda(), torch.from_numpy(
        dir_ret_np).cuda(), torch.from_numpy(
            labels_ret_np).cuda(), torch.from_numpy(scores_ret_np).cuda()


def rotate_weighted_nms_cc(
    box,
    dets,
    thresh,
    iou_preds,
    labels,
    dirs,
    anchors=None,
):
    scores = dets[:, 5]
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2:4],
                                                     dets[:, 4])
    dets_standup = box_np_ops.corner_to_standup_nd_jit(dets_corners)
    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)

    result = IOU_weighted_rotate_non_max_suppression_cpu(
        box, dets_corners, standup_iou, thresh, scores, iou_preds, labels,
        dirs, anchors)

    return result
