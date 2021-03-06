_base_ = [
    '../_base_/models/hv_second_secfpn_kitti.py',
    '../_base_/datasets/kitti-3d-3class.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(
    use_iou_regressor=True,
    loss_iou=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
))
#  data = dict(samples_per_gpu=12, workers_per_gpu=4)
data = dict(samples_per_gpu=4, workers_per_gpu=2)
evaluation = dict(interval=40)
