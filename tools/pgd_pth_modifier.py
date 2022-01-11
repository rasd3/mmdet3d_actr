import torch
import copy

model_ori = torch.load(
    './model_zoo/pgd_r101_caffe_fpn_gn-head_3x4_4x_kitti-mono3d_20211022_102608-8a97533b.pth'
)
model = copy.deepcopy(model_ori)

for key in model['state_dict'].copy().keys():
    if 'backbone' in key or 'neck' in key:
        model['state_dict']['img_' + key] = model['state_dict'][key]
    model['state_dict'].pop(key)

torch.save(model, './model_zoo/pgd_r101_caffe_fpn_gn_img_backbone.pth')
