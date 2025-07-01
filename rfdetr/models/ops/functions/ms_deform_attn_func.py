# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
ms_deform_attn_func
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """"for debug and test only, need to use cuda version instead
    """
    # B, n_heads, head_dim, N
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, n_heads_, L, P, _ = sampling_locations.shape
    assert n_heads_ == n_heads

    # Precompute sizes and indices
    device = value.device
    num_levels = value_spatial_shapes.shape[0]
    spatial_nums = [H * W for H, W in value_spatial_shapes]
    spatial_cumsum = [0]
    for n in spatial_nums:
        spatial_cumsum.append(spatial_cumsum[-1] + n)

    # Avoid split: slice view directly where needed
    sampling_grids = 2 * sampling_locations - 1
    # Preallocate result list for sampling values
    # To avoid torch.stack overhead + flatten(-2) later,
    # we'll preallocate a tensor of [B*n_heads, head_dim, Len_q, num_levels*P] and assign each slice
    sampling_value_tensor = value.new_zeros((B * n_heads, head_dim, Len_q, num_levels * P))

    for lid_, (H, W) in enumerate(value_spatial_shapes.tolist()):
        start = spatial_cumsum[lid_]
        end = spatial_cumsum[lid_ + 1]
        # B, n_heads, head_dim, H, W
        value_l_ = value[:, :, :, start:end].view(B * n_heads, head_dim, H, W)
        # B, Len_q, n_heads, P, 2 -> B, n_heads, Len_q, P, 2 -> B*n_heads, Len_q, P, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).reshape(B * n_heads, Len_q, P, 2)
        # B*n_heads, head_dim, Len_q, P
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        # Assign directly into preallocated tensor
        # sampling_value_l_: [B*n_heads, head_dim, Len_q, P]
        sampling_value_tensor[:, :, :, lid_*P:(lid_+1)*P] = sampling_value_l_

    # (B, Len_q, n_heads, L * P) -> (B, n_heads, Len_q, L, P) -> (B*n_heads, 1, Len_q, L*P)
    attention_weights = attention_weights.transpose(1, 2).contiguous().view(B * n_heads, 1, Len_q, L * P)
    # B*n_heads, head_dim, Len_q, L*P
    sampling_value_flat = sampling_value_tensor   # already [B*n_heads, head_dim, Len_q, L*P]

    # Avoid: stack + flatten(-2)
    # output = (sampling_value_list * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    output = (sampling_value_flat * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()
