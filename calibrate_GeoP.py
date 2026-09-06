import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import cv2
import json
import numpy as np
import pytorch3d.transforms
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, weighted_l1_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, CalibrateGaussianListModel
from utils.general_utils import safe_state, get_expon_lr_func, sample_uniform
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from visualizer.pose_visualizer import *
from visualizer.image_visualizer import project_lidar_to_image_with_projection, \
    project_to_pixel_torch, get_3d_points_from_pixels_depth_mask_torch
import torch.nn.functional as F
import time
from datetime import datetime


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def has_nan(x):
    if x is None:
        return False
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()


def safe_inverse(x: torch.Tensor, eps: float = 1e-6, max_value: float = 1e6) -> torch.Tensor:
    x_safe = torch.nan_to_num(x, nan=eps, posinf=eps, neginf=eps)
    x_safe = torch.clamp(x_safe, min=eps)
    inv = 1.0 / x_safe
    return torch.clamp(inv, max=max_value)


def normalize_quaternion_(q: torch.Tensor, eps: float = 1e-8) -> None:
    with torch.no_grad():
        q.data = torch.nan_to_num(q.data, nan=0.0, posinf=0.0, neginf=0.0)
        norm = torch.linalg.norm(q.data)
        if (not torch.isfinite(norm)) or norm < eps:
            q.data.zero_()
            q.data[0] = 1.0
        else:
            q.data /= norm


def sanitize_extrinsic_grads(scene, clip_norm: float = 1.0) -> bool:
    rot_param = scene.sensor_trajectory.rotation_cl_delta
    trans_param = scene.sensor_trajectory.translation_cl_delta
    had_nonfinite = False

    for p in (rot_param, trans_param):
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            had_nonfinite = True

    params_with_grad = [p for p in (rot_param, trans_param) if p.grad is not None]
    if params_with_grad:
        torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=clip_norm)

    return had_nonfinite


def safe_project_to_pixel_torch(points_3d: torch.Tensor, P: torch.Tensor, image_width: int, image_height: int, w_eps: float = 1e-6) -> torch.Tensor:
    ones = torch.ones_like(points_3d[:, 0:1])
    points_homo = torch.hstack([points_3d, ones])
    clip_coords = points_homo @ P.T
    w = clip_coords[:, 3:4]
    w_sign = torch.sign(w)
    w_sign[w_sign == 0] = 1.0
    w_safe = torch.where(torch.abs(w) < w_eps, w_sign * w_eps, w)
    ndc_coords = clip_coords[:, :3] / w_safe
    u = (ndc_coords[:, 0] + 1) * 0.5 * image_width
    v = (ndc_coords[:, 1] + 1) * 0.5 * image_height
    return torch.column_stack([u, v])


def nan_report(tag: str, **tensors):
    bad = {k: v for k, v in tensors.items() if has_nan(v)}
    if bad:
        names = ", ".join(bad.keys())
        print(f"[NaN/Inf DETECTED] {tag}: {names}")


def dilate_false_mask(mask: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size % 2 == 0:
        raise ValueError("window_size must be an odd number, e.g., 3,5,7...")

    assert mask.dtype == torch.bool, "Input mask must be a boolean tensor."

    inverted_mask = ~mask
    inverted_mask_float = inverted_mask.float()

    kernel = torch.ones(
        (1, 1, window_size, window_size),
        dtype=torch.float32,
        device=mask.device
    )

    shape = inverted_mask_float.shape
    if inverted_mask_float.dim() == 2:  # (H, W) � (1, 1, H, W)
        inverted_mask_float = inverted_mask_float.view(1, 1, shape[0], shape[1])
    elif inverted_mask_float.dim() == 3:  # (C, H, W) � (1, C, H, W)
        inverted_mask_float = inverted_mask_float.unsqueeze(0)

    padding = window_size // 2
    conved = F.conv2d(inverted_mask_float, kernel, padding=padding)

    dilated_inverted = conved > 0

    dilated_inverted = dilated_inverted.view(shape)

    result = ~dilated_inverted
    return result


def compute_rgb_projection_error(
        rgb_render,
        rgb_target,
        depth_render,
        depth_mask,
        gradient_mask,
        points3d_world,
        rotation_wc_ref,
        translation_wc_ref,
        rotation_wc_target,
        translation_wc_target,
        P,
        target_depth = None):
    img_height, img_width = rgb_render.shape[1],rgb_render.shape[2]
    valid_flat_indices = torch.where(depth_mask.flatten())[0]

    points3d_target = points3d_world @ rotation_wc_target  - (rotation_wc_target.T @ translation_wc_target)
    project_depth = points3d_target[:,2]
    finite_depth = torch.isfinite(project_depth)
    if finite_depth.sum() == 0:
        return torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
    points3d_target = points3d_target[finite_depth]
    project_depth = project_depth[finite_depth]
    valid_flat_indices = valid_flat_indices[finite_depth]

    target2d = safe_project_to_pixel_torch(points3d_target, P, img_width, img_height)
    finite_target2d = torch.isfinite(target2d).all(dim=1)
    if finite_target2d.sum() == 0:
        return torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
    target2d = target2d[finite_target2d]
    project_depth = project_depth[finite_target2d]
    valid_flat_indices = valid_flat_indices[finite_target2d]
    us,vs = target2d[:,0], target2d[:,1]
    projection_mask = (target2d[:,0] >= 0.) & (target2d[:,0] < img_width) & (target2d[:,1] >= 0.) & (target2d[:,1] < img_height)
    if projection_mask.sum() == 0:
        return torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
    us = us[projection_mask]/img_width
    vs = vs[projection_mask]/img_height
    grid_x = us * 2 - 1 
    grid_y = vs * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)  
    grid = grid.view(1, 1, -1, 2)


    sampled = F.grid_sample(
        rgb_target.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)  # (1, C, 1, N)
    sampled_depth = F.grid_sample(
        target_depth.unsqueeze(0),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)
    #Inverse depth
    sampled_depth = torch.clamp(torch.nan_to_num(sampled_depth, nan=1e7, posinf=1e7, neginf=1e7), min=1e-4)
    project_depth_diff = project_depth[projection_mask] - safe_inverse(sampled_depth.squeeze(0), eps=1e-4, max_value=1e4)
    project_depth_diff = torch.nan_to_num(project_depth_diff, nan=1e7, posinf=1e7, neginf=-1e7)
    depth_culling = ((project_depth_diff) < 0.2)
    indices_true = valid_flat_indices[projection_mask][depth_culling]

    if indices_true.numel() == 0:
        return torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)

    sampled = sampled[:,depth_culling]
    initial_rgb = rgb_render.view(3,-1)[:,indices_true]
    gradient_flat = gradient_mask.view(gradient_mask.shape[0], -1)
    weights = (2 - gradient_flat[:, indices_true]).detach()
    residual = torch.nan_to_num(sampled - initial_rgb, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.mean(weights * torch.abs(residual))


def render_rgb_projection(
        rgb_render,
        rgb_target,
        depth_render,
        depth_mask,
        gradient_mask,
        points3d_world,
        rotation_wc_ref,
        translation_wc_ref,
        rotation_wc_target,
        translation_wc_target,
        P,
        target_depth = None):
    img_height, img_width = rgb_render.shape[1],rgb_render.shape[2]
    valid_flat_indices = torch.where(depth_mask.flatten())[0]

    points3d_target = points3d_world @ rotation_wc_target  - (rotation_wc_target.T @ translation_wc_target)
    project_depth = points3d_target[:,2]
    finite_depth = torch.isfinite(project_depth)
    if finite_depth.sum() == 0:
        return torch.zeros_like(rgb_render), torch.zeros_like(rgb_render)
    points3d_target = points3d_target[finite_depth]
    project_depth = project_depth[finite_depth]
    valid_flat_indices = valid_flat_indices[finite_depth]

    target2d = safe_project_to_pixel_torch(points3d_target, P, img_width, img_height)
    finite_target2d = torch.isfinite(target2d).all(dim=1)
    if finite_target2d.sum() == 0:
        return torch.zeros_like(rgb_render), torch.zeros_like(rgb_render)
    target2d = target2d[finite_target2d]
    project_depth = project_depth[finite_target2d]
    valid_flat_indices = valid_flat_indices[finite_target2d]
    us,vs = target2d[:,0], target2d[:,1]
    projection_mask = (target2d[:,0] >= 0.) & (target2d[:,0] < img_width) & (target2d[:,1] >= 0.) & (target2d[:,1] < img_height)
    if projection_mask.sum() == 0:
        return torch.zeros_like(rgb_render), torch.zeros_like(rgb_render)
    us = us[projection_mask]/img_width
    vs = vs[projection_mask]/img_height
    grid_x = us * 2 - 1  # [0,1] � [-1,1]
    grid_y = vs * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, 2)
    grid = grid.view(1, 1, -1, 2)


    sampled = F.grid_sample(
        rgb_target.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)  # (1, C, 1, N)
    sampled_depth = F.grid_sample(
        target_depth.unsqueeze(0),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)
    #Inverse depth
    sampled_depth = torch.clamp(torch.nan_to_num(sampled_depth, nan=1e7, posinf=1e7, neginf=1e7), min=1e-4)
    project_depth_diff = project_depth[projection_mask] - safe_inverse(sampled_depth.squeeze(0), eps=1e-4, max_value=1e4)
    project_depth_diff = torch.nan_to_num(project_depth_diff, nan=1e7, posinf=1e7, neginf=-1e7)
    depth_culling = ((project_depth_diff) < 0.2)
    indices_true = valid_flat_indices[projection_mask][depth_culling]

    if indices_true.numel() == 0:
        return torch.zeros_like(rgb_render), torch.zeros_like(rgb_render)

    sampled = sampled[:,depth_culling]
    initial_rgb = rgb_render.view(3,-1)[:,indices_true]
    gradient_flat = gradient_mask.view(gradient_mask.shape[0], -1)
    weights = (2 - gradient_flat[:, indices_true]).detach()

    # ===== 추가된 부분 =====
    C, H, W = rgb_render.shape

    # 빈 canvas 생성
    initial_rgb_img = torch.zeros_like(rgb_render)   # (3, H, W)
    sampled_rgb_img = torch.zeros_like(rgb_render)   # (3, H, W)

    # flat view
    initial_rgb_flat = initial_rgb_img.view(3, -1)
    sampled_rgb_flat = sampled_rgb_img.view(3, -1)

    # scatter
    initial_rgb_flat[:, indices_true] = initial_rgb
    sampled_rgb_flat[:, indices_true] = sampled
    # ======================

    return initial_rgb_img, sampled_rgb_img



def compute_rgb_projection_error_and_flow_error(
        rgb_render,
        rgb_target,
        depth_render,
        depth_mask,
        gradient_mask,
        points3d_world,
        rotation_wc_ref,
        translation_wc_ref,
        rotation_wc_target,
        translation_wc_target,
        P,
        flow,
        flow_certainties,
        visualize = True,
        target_depth = None
        ):
    img_height, img_width = rgb_render.shape[1],rgb_render.shape[2]
    valid_flat_indices = torch.where(depth_mask.flatten())[0]


    points3d_target = points3d_world @ rotation_wc_target  - (rotation_wc_target.T @ translation_wc_target)
    project_depth = points3d_target[:, 2]
    finite_depth = torch.isfinite(project_depth)
    if finite_depth.sum() == 0:
        zero = torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
        return zero, zero
    points3d_target = points3d_target[finite_depth]
    project_depth = project_depth[finite_depth]
    valid_flat_indices = valid_flat_indices[finite_depth]

    target2d = safe_project_to_pixel_torch(points3d_target, P, img_width, img_height)
    finite_target2d = torch.isfinite(target2d).all(dim=1)
    if finite_target2d.sum() == 0:
        zero = torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
        return zero, zero
    target2d = target2d[finite_target2d]
    project_depth = project_depth[finite_target2d]
    valid_flat_indices = valid_flat_indices[finite_target2d]
    us,vs = target2d[:,0], target2d[:,1]
    projection_mask = (target2d[:,0] >= 0.) & (target2d[:,0] < img_width) & (target2d[:,1] >= 0.) & (target2d[:,1] < img_height)
    if projection_mask.sum() == 0:
        zero = torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
        return zero, zero
    us = us[projection_mask]/img_width
    vs = vs[projection_mask]/img_height
    grid_x = us * 2 - 1  
    grid_y = vs * 2 - 1

    projection_pos = torch.concatenate([grid_x[:,None], grid_y[:,None]], dim=1)


    #Compute flow error
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, 2)
    grid = grid.view(1, 1, -1, 2)

    sampled = F.grid_sample(
        rgb_target.unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)  # (1, C, 1, N)
    sampled_depth = F.grid_sample(
        target_depth.unsqueeze(0),
        grid,
        mode='nearest',
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(1)

    sampled_depth = torch.clamp(torch.nan_to_num(sampled_depth, nan=1e7, posinf=1e7, neginf=1e7), min=1e-4)

    project_depth_proj = project_depth[projection_mask]
    inv_sampled = safe_inverse(sampled_depth.squeeze(0), eps=1e-4, max_value=1e4)
    project_depth_diff = project_depth_proj - inv_sampled
    project_depth_diff = torch.nan_to_num(project_depth_diff, nan=1e7, posinf=1e7, neginf=-1e7)
    depth_culling = ((project_depth_diff) < 0.2)

    indices_true = valid_flat_indices[projection_mask][depth_culling]
    if indices_true.numel() == 0:
        zero = torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
        return zero, zero

    flow_pos = flow.view(-1,2)[indices_true]

    flow_mask = (flow_certainties.view(-1)[indices_true] > 0.5) 

    if flow_mask.sum() == 0:
        flow_loss = torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
    else:
        proj_pos_culled = projection_pos[depth_culling] 
        flow_diff = flow_pos[flow_mask] - proj_pos_culled[flow_mask]
        flow_diff = torch.nan_to_num(flow_diff, nan=0.0, posinf=0.0, neginf=0.0)
        flow_loss = torch.mean(torch.abs(flow_diff))

    sampled = sampled[:,depth_culling]

    initial_rgb = rgb_render.view(3, -1)[:, indices_true]
    gradient_flat = gradient_mask.view(gradient_mask.shape[0], -1)
    weights = (2 - gradient_flat[:, indices_true]).detach()

    if sampled.numel() == 0 or initial_rgb.numel() == 0:
        match_loss = torch.zeros([], device=rgb_render.device, dtype=rgb_render.dtype)
    else:
        residual = torch.nan_to_num(sampled - initial_rgb, nan=0.0, posinf=0.0, neginf=0.0)
        match_loss = torch.mean(weights * torch.abs(residual))

    if not torch.isfinite(match_loss):
        print("match_loss is not finite")
    if not torch.isfinite(flow_loss):
        print("flow_loss is not finite")

    return match_loss, flow_loss


def compute_depth_error(point_cloud_lidar, scene, P, depth_render, depth_mask):
    rotation_cl, translation_cl = scene.sensor_trajectory.get_extrinsics()
    rotation_cl = rotation_cl.detach()
    rotation_cl = pytorch3d.transforms.quaternion_to_matrix(rotation_cl)
    translation_cl = translation_cl.detach() * 0.
    point_cloud_camera = point_cloud_lidar @ rotation_cl.T + translation_cl
    point_cloud_depths = point_cloud_camera[:, 2]
    width = depth_render.shape[2]
    height = depth_render.shape[1]
    point_cloud_pixels = project_to_pixel_torch(point_cloud_camera, P, width, height)
    u, v = point_cloud_pixels[:, 0], point_cloud_pixels[:, 1]
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (point_cloud_depths > 0.1) & (
                point_cloud_depths < 100.)
    if valid_uv.sum() == 0:
        return torch.zeros([], device=depth_render.device, dtype=depth_render.dtype)
    depth_render_view = depth_render.view(1, 1, height, width)
    grid_u = (u / (width - 1)) * 2 - 1  
    grid_v = (v / (height - 1)) * 2 - 1  
    grid = torch.stack([grid_u[valid_uv], grid_v[valid_uv]], dim=1).view(1, -1, 1, 2)  # [1,N,1,2]
    depth_sampled = F.grid_sample(depth_render_view, grid, mode='nearest', align_corners=True).view(-1)
    depth_mask_loss = F.grid_sample(depth_mask.float().unsqueeze(0), grid, mode='nearest', align_corners=True).view(-1)
    if depth_sampled.numel() == 0:
        return torch.zeros([], device=depth_render.device, dtype=depth_render.dtype)
    depth_sampled = torch.nan_to_num(depth_sampled, nan=0.0, posinf=0.0, neginf=0.0)
    depth_target = safe_inverse(point_cloud_depths[valid_uv], eps=1e-4, max_value=1e4)
    depth_loss = l1_loss(depth_sampled * depth_mask_loss, depth_target * depth_mask_loss)
    if not torch.isfinite(depth_loss):
        return torch.zeros([], device=depth_render.device, dtype=depth_render.dtype)
    return depth_loss


# inverse_version
def compute_dense_depth_error(point_cloud_lidar, scene, P, depth_render, depth_mask, depth_ratio = 0.05, temperature = 0.01):

    rotation_cl, translation_cl = scene.sensor_trajectory.get_extrinsics()
    rotation_cl = pytorch3d.transforms.quaternion_to_matrix(rotation_cl.detach())
    translation_cl = translation_cl.detach() * 0.
    point_cloud_camera = point_cloud_lidar @ rotation_cl.T + translation_cl
    point_cloud_depths = point_cloud_camera[:, 2]
    width = depth_render.shape[2]
    height = depth_render.shape[1]
    point_cloud_pixels = project_to_pixel_torch(point_cloud_camera, P, width, height)
    u, v = point_cloud_pixels[:, 0], point_cloud_pixels[:, 1]
    valid_uv = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height) &
        (point_cloud_depths > 0.1) &
        (point_cloud_depths < 100.)
    )
    if valid_uv.sum() == 0:
        return torch.zeros([], device=depth_render.device, dtype=depth_render.dtype)
    u = u[valid_uv]
    v = v[valid_uv]
    lidar_depth = point_cloud_depths[valid_uv]
    lidar_inv_depth = safe_inverse(lidar_depth, eps=1e-6, max_value=1e4)
    depth_render_view = depth_render.view(1, 1, height, width)
    grid_u = (u / (width - 1)) * 2 - 1
    grid_v = (v / (height - 1)) * 2 - 1
    grid = torch.stack([grid_u, grid_v], dim=1).view(1, -1, 1, 2)
    depth_sampled = F.grid_sample(
        depth_render_view, grid,
        mode='nearest', align_corners=True
    ).view(-1)
    depth_mask_loss = F.grid_sample(
        depth_mask.float().unsqueeze(0),
        grid,
        mode='nearest', align_corners=True
    ).view(-1)
    if depth_sampled.numel() == 0:
        return torch.zeros([], device=depth_render.device, dtype=depth_render.dtype)

    render_inv_depth = torch.nan_to_num(depth_sampled, nan=0.0, posinf=0.0, neginf=0.0)
    render_depth = safe_inverse(render_inv_depth, eps=1e-6, max_value=1e4)

    diff_linear = (render_depth * (1.0 + depth_ratio) ) - lidar_depth
    visible_weight = torch.sigmoid(diff_linear / temperature)

    final_weight = depth_mask_loss * visible_weight
    abs_diff_inv = torch.abs(render_inv_depth - lidar_inv_depth)
    
    depth_loss = (abs_diff_inv * final_weight).sum() / (final_weight.sum() + 1e-6)
    if not torch.isfinite(depth_loss):
        return torch.zeros([], device=depth_render.device, dtype=depth_render.dtype)

    return depth_loss




def render_depth_error(point_cloud_lidar, scene, P, depth_render, depth_mask):
    rotation_cl, translation_cl = scene.sensor_trajectory.get_extrinsics()
    rotation_cl = rotation_cl.detach()
    rotation_cl = pytorch3d.transforms.quaternion_to_matrix(rotation_cl)
    translation_cl = translation_cl.detach() * 0.
    point_cloud_camera = point_cloud_lidar @ rotation_cl.T + translation_cl
    point_cloud_depths = point_cloud_camera[:, 2]
    width = depth_render.shape[2]
    height = depth_render.shape[1]
    pc = point_cloud_camera
    z = pc[:, 2]

    valid = z > 0.1
    pc = pc[valid]
    z = z[valid]

    uv = project_to_pixel_torch(pc, P, width, height)
    u = uv[:, 0].long()
    v = uv[:, 1].long()

    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]

    depth_map = torch.full((height, width), float('inf'), device=z.device)

    idx = v * width + u

    depth_map.view(-1).scatter_reduce_(
        0, idx, z, reduce="amin", include_self=True
    )

    depth_map[depth_map == float('inf')] = 0.0

    return depth_map



def compute_match_loss(scene, vind, all_depths, all_depth_masks, resolution_scale, gt_image, gt_gradient_mask, gt_pixel_mask, rendered_depth, depth_mask, window_size = 2, use_flow = False, visualize = False):
    match_loss = torch.zeros([], device=gt_image.device, dtype=gt_image.dtype)
    loss_num = 0
    rotation_wc_ref, translation_wc_ref = scene.sensor_trajectory.get_camera_pose(vind)
    cam_num = len(scene.getTrainCameras())
    shift_indices = [i for i in range(-window_size, window_size+1) if vind+i >=0 and vind+i<cam_num and i != 0]
    ref_cam = scene.getTrainCameraByIndex(vind, scale=resolution_scale)

    for shift in shift_indices:
        target_vind = vind + shift
        target_cam = scene.getTrainCameraByIndex(target_vind, scale=resolution_scale)
        depth_P = target_cam.projection_matrix.transpose(0, 1)
        rotation_wc_match, translation_wc_match = scene.sensor_trajectory.get_camera_pose(target_vind)
        rgb_target = target_cam.original_image.cuda()


        with torch.no_grad():
            target_depth = all_depths[target_vind]
            target_depth_mask = all_depth_masks[target_vind]
            target_depth[target_depth_mask == False] = 1e7

        flow = None
        flow_certainties = None
        if use_flow:
            for i in range(len(ref_cam.forward_flow_uids)):
                if target_vind == ref_cam.forward_flow_uids[i]:
                    flow = ref_cam.forward_flows[i]
                    flow_certainties = ref_cam.forward_certainties[i]
            for i in range(len(ref_cam.backward_flow_uids)):
                if target_vind == ref_cam.backward_flow_uids[i]:
                    flow = ref_cam.backward_flows[i]
                    flow_certainties = ref_cam.backward_certainties[i]

        img_height, img_width = gt_image.shape[1], gt_image.shape[2]
        points3d = get_3d_points_from_pixels_depth_mask_torch(
            depth_map=safe_inverse(rendered_depth.squeeze(0), eps=1e-4, max_value=1e4),
            mask=depth_mask.squeeze(0),
            P=depth_P,
            image_width=img_width,
            image_height=img_height)
        if points3d.numel() == 0:
            continue
        points3d_world = points3d @ rotation_wc_ref.T + translation_wc_ref

        if flow is None:
            rgb_rep_loss = compute_rgb_projection_error(
                rgb_render=gt_image,
                rgb_target=rgb_target,
                depth_render=rendered_depth,
                depth_mask=depth_mask,
                gradient_mask = gt_gradient_mask.unsqueeze(0),
                points3d_world=points3d_world,
                rotation_wc_ref=rotation_wc_ref,
                translation_wc_ref=translation_wc_ref,
                rotation_wc_target=rotation_wc_match,
                translation_wc_target=translation_wc_match,
                P=depth_P,
                target_depth=target_depth)
            flow_loss = None
        else:
            rgb_rep_loss, flow_loss = compute_rgb_projection_error_and_flow_error(
                rgb_render=gt_image,
                rgb_target=rgb_target,
                depth_render=rendered_depth,
                depth_mask=depth_mask,
                gradient_mask = gt_gradient_mask.unsqueeze(0),
                points3d_world=points3d_world,
                rotation_wc_ref=rotation_wc_ref,
                translation_wc_ref=translation_wc_ref,
                rotation_wc_target=rotation_wc_match,
                translation_wc_target=translation_wc_match,
                P=depth_P,
                flow=flow,
                flow_certainties=flow_certainties,
                visualize = visualize,
                target_depth=target_depth
            )
        match_loss += rgb_rep_loss
        if flow_loss is not None:
            match_loss += flow_loss
        loss_num += 1

    if loss_num == 0:
        return torch.zeros([], device=gt_image.device, dtype=gt_image.dtype)

    match_loss /= loss_num


    return match_loss


def get_reproj_images(scene, vind, all_depths, all_depth_masks, resolution_scale, gt_image, gt_gradient_mask, gt_pixel_mask, rendered_depth, depth_mask, window_size = 2, use_flow = False, visualize = False):
    match_loss = 0.
    loss_num = 0
    rotation_wc_ref, translation_wc_ref = scene.sensor_trajectory.get_camera_pose(vind)
    cam_num = len(scene.getTrainCameras())
    shift_indices = [i for i in range(-window_size, window_size+1) if vind+i >=0 and vind+i<cam_num and i != 0]
    ref_cam = scene.getTrainCameraByIndex(vind, scale=resolution_scale)

    for shift in shift_indices:
        target_vind = vind + shift
        target_cam = scene.getTrainCameraByIndex(target_vind, scale=resolution_scale)
        depth_P = target_cam.projection_matrix.transpose(0, 1)
        rotation_wc_match, translation_wc_match = scene.sensor_trajectory.get_camera_pose(target_vind)
        rgb_target = target_cam.original_image.cuda()


        with torch.no_grad():
            target_depth = all_depths[target_vind]
            target_depth_mask = all_depth_masks[target_vind]
            target_depth[target_depth_mask == False] = 1e7

        flow = None
        flow_certainties = None
        if use_flow:
            for i in range(len(ref_cam.forward_flow_uids)):
                if target_vind == ref_cam.forward_flow_uids[i]:
                    flow = ref_cam.forward_flows[i]
                    flow_certainties = ref_cam.forward_certainties[i]
            for i in range(len(ref_cam.backward_flow_uids)):
                if target_vind == ref_cam.backward_flow_uids[i]:
                    flow = ref_cam.backward_flows[i]
                    flow_certainties = ref_cam.backward_certainties[i]

        img_height, img_width = gt_image.shape[1], gt_image.shape[2]
        points3d = get_3d_points_from_pixels_depth_mask_torch(
            depth_map=safe_inverse(rendered_depth.squeeze(0), eps=1e-4, max_value=1e4),
            mask=depth_mask.squeeze(0),
            P=depth_P,
            image_width=img_width,
            image_height=img_height)
        if points3d.numel() == 0:
            continue

        points3d_world = points3d @ rotation_wc_ref.T + translation_wc_ref

        origin_img, reproj_img = render_rgb_projection(
            rgb_render=gt_image,
            rgb_target=rgb_target,
            depth_render=rendered_depth,
            depth_mask=depth_mask,
            gradient_mask = gt_gradient_mask.unsqueeze(0),
            points3d_world=points3d_world,
            rotation_wc_ref=rotation_wc_ref,
            translation_wc_ref=translation_wc_ref,
            rotation_wc_target=rotation_wc_match,
            translation_wc_target=translation_wc_match,
            P=depth_P,
            target_depth=target_depth)

        return origin_img, reproj_img

    return torch.zeros_like(gt_image), torch.zeros_like(gt_image)


def generate_all_depths(scene, pipe, background, resolution_scale, train_test_exp, full_cut_size = 100, min_depth = 0.1, max_depth = 50.):
    all_depths = []
    all_depth_masks = []
    with torch.no_grad():
        train_cameras = scene.getTrainCameras(scale=resolution_scale).copy()
        cam_num = len(train_cameras)
        for i in range(cam_num):
            target_vind = i
            target_cam = train_cameras[target_vind]
            rotation_wc_match, translation_wc_match = scene.sensor_trajectory.get_camera_pose(target_vind)
            render_pkg = render(target_cam, scene.gaussians, pipe, background, use_trained_exp= train_test_exp,
                                    separate_sh=SPARSE_ADAM_AVAILABLE, cam_rotation=rotation_wc_match,
                                    cam_translation=translation_wc_match)
            target_depth = render_pkg['depth'].detach()

            target_depth_mask = torch.ones_like(target_depth).bool().detach()
            cut_region = int(full_cut_size / resolution_scale)
            target_depth_mask[:, :cut_region, :] = False
            inv_depth = safe_inverse(target_depth, eps=1e-4, max_value=1e4)
            target_depth_mask[inv_depth > max_depth] = False
            target_depth_mask[inv_depth < min_depth] = False
            all_depths.append(target_depth)
            all_depth_masks.append(target_depth_mask)
    return all_depths, all_depth_masks


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_path):
    vox_resol = opt.voxel_res
    # TODO 로그 이름 변경
    tb_name = f"test_GeoP_sequence_{dataset.data_seq}_camid_{dataset.cam_id}_resolution_{vox_resol}"
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"results/runs_GeoP/{tb_name}_{current_time}" 
    if log_path != None:
        log_dir = f"results/runs_GeoP_{log_path}/{tb_name}_{current_time}" 
    os.makedirs(log_dir, exist_ok=True)

    writer = None
    if TENSORBOARD_FOUND:
        writer = SummaryWriter(log_dir)
        print(f"--- TensorBoard 로깅을 시작합니다. 로그 디렉토리: {log_dir} ---")
    
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    resolution_scales = [4., 2., 1., 1.]
    voxel_resolutions = [vox_resol, vox_resol, vox_resol, vox_resol]
    initial_resolution = resolution_scales[0]
    change_iteration = [0, 2000, 4000, 8000]
    per_level_iterations = [2000, 2000, 4000, 7000]
    lr_ratio = [1., 0.5, 0.5, 0.1]
    current_scale_index = 0
    match_loss_window_size = 2
    fast_mode = False

    gaussians = CalibrateGaussianListModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, resolution_scales=resolution_scales, voxel_resolutions=voxel_resolutions)
    gaussians.training_setup(opt)
    scene.sensor_trajectory.training_setup(opt)


    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = np.random.rand(3)
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    viewpoint_stack = scene.getTrainCameras(initial_resolution).copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    cut_full_size = 10
    min_depth = 0.1
    max_depth = 50.

    depth_view = None
    #Prepare depth maps for latter
    all_depths, all_depth_masks = generate_all_depths(scene, pipe, background, resolution_scales[current_scale_index], train_test_exp=dataset.train_test_exp, full_cut_size = cut_full_size, min_depth = min_depth, max_depth = max_depth)

    sum_iterations = 10000
    if fast_mode:
        sum_iterations = 6000

    last_check_iter = 0
    Nan_detected = 0

    for iteration in range(first_iter, sum_iterations):
        # Manage checkpoints
        if last_check_iter + 100 <= iteration:
            if Nan_detected > 0:
                Nan_detected -= 1
                last_check_iter = iteration
                continue
            with torch.no_grad():
                current_rot = scene.sensor_trajectory.rotation_cl.clone().detach()
                current_trans = scene.sensor_trajectory.translation_cl.clone().detach()

                last_check_iter = iteration

        start_time = time.perf_counter()
        if network_gui.conn == None:
            network_gui.try_connect()

        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,
                                       use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        p_updated = False


        # TODO:Not need now

        if iteration >= change_iteration[1] and current_scale_index == 0:
            print('Change to pyramid level 1')
            current_scale_index = 1
            scene.update_level(current_scale_index)
            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background, resolution_scales[current_scale_index], train_test_exp=dataset.train_test_exp, full_cut_size = cut_full_size, min_depth = min_depth, max_depth = max_depth)
            viewpoint_stack.clear()
            viewpoint_indices.clear()
            scene.sensor_trajectory.update_learning_rate(lr_ratio[current_scale_index])
            print('Current point size: ', len(gaussians._xyz))
        elif iteration >= change_iteration[2] and current_scale_index == 1:
            print('Change to pyramid level 2')
            current_scale_index = 2
            scene.update_level(current_scale_index)
            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background,
                                                              resolution_scales[current_scale_index],
                                                              train_test_exp=dataset.train_test_exp,
                                                              full_cut_size=cut_full_size, min_depth=min_depth,
                                                              max_depth=max_depth)
            viewpoint_stack.clear()
            viewpoint_indices.clear()
            scene.sensor_trajectory.update_learning_rate(lr_ratio[current_scale_index])
            print('Current point size: ', len(gaussians._xyz))
        elif iteration >= change_iteration[3] and current_scale_index == 2:
            print('Change to pyramid level 3')
            current_scale_index = 3
            scene.update_level(current_scale_index)
            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background,
                                                              resolution_scales[current_scale_index],
                                                              train_test_exp=dataset.train_test_exp,
                                                              full_cut_size=cut_full_size, min_depth=min_depth,
                                                              max_depth=max_depth)
            viewpoint_stack.clear()
            viewpoint_indices.clear()
            scene.sensor_trajectory.update_learning_rate(lr_ratio[current_scale_index])
            print('Current point size: ', len(gaussians._xyz))

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale=resolution_scales[current_scale_index]).copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        cam_rotation, cam_translation = scene.sensor_trajectory.get_camera_pose(vind)


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE, cam_rotation=cam_rotation,
                            cam_translation=cam_translation)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg['depth']

        end_time_render = time.perf_counter()
        execution_time_render = end_time_render - start_time

        depth_mask = torch.ones_like(rendered_depth).bool().detach()
        cut_region = int(cut_full_size / resolution_scales[current_scale_index])
        depth_mask[:, :cut_region, :] = False
        image_mask = (depth_mask.repeat(3, 1, 1)).detach()
        inv_depth = safe_inverse(rendered_depth.detach(), eps=1e-4, max_value=1e4)
        depth_mask[inv_depth > max_depth] = False
        depth_mask[inv_depth < min_depth] = False

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_gradient_mask = viewpoint_cam.gradient_mask.cuda()
        gt_pixel_mask = viewpoint_cam.pixel_mask.cuda()
        weights = gt_gradient_mask.unsqueeze(0).repeat(3,1,1)[image_mask]
        if scene.sensor_trajectory.rotation_cl_delta.requires_grad:
            Ll1 = weighted_l1_loss(image[image_mask], gt_image[image_mask], 2-weights)
        else:
            Ll1 = l1_loss(image[image_mask], gt_image[image_mask])

        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image, window_size=5)

        rotation_wl_list, translations_wl_list = scene.sensor_trajectory.get_all_lidar_poses()
        point_cloud_lidar = gaussians.gaussian_model_list[vind].original_point_cloud
        point_cloud_world = gaussians.original_point_cloud
        rotation_wl_vind = pytorch3d.transforms.quaternion_to_matrix(rotation_wl_list[vind].detach())
        translation_wl_vind = translations_wl_list[vind].detach()
        point_cloud_lidar_dense = (point_cloud_world - translation_wl_vind) @ rotation_wl_vind


        depth_P = viewpoint_cam.projection_matrix.transpose(0, 1)


        end_time_loss_render = time.perf_counter()
        execution_time_loss_render = end_time_loss_render - start_time

        _, lidar_translation = scene.sensor_trajectory.get_lidar_pose(vind)
        depth_cam_rotation = cam_rotation.detach()
        depth_lidar_translation = lidar_translation.detach()
        render_pkg_depth = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp,
                              separate_sh=SPARSE_ADAM_AVAILABLE, cam_rotation= depth_cam_rotation,
                              cam_translation=depth_lidar_translation)

        depth_mask2 = torch.ones_like(render_pkg_depth['depth']).bool().detach()
        depth_mask2[:, :cut_region, :] = False
        inv_depth2 = safe_inverse(render_pkg_depth['depth'].detach(), eps=1e-4, max_value=1e4)
        depth_mask2[inv_depth2 > max_depth] = False
        depth_mask2[inv_depth2 < min_depth] = False
        depth_loss = compute_depth_error(point_cloud_lidar, scene, depth_P, render_pkg_depth['depth'],
                                             depth_mask2.detach())
        
        depth_loss_dense = 0.
        if current_scale_index >= opt.dense_stage: 
            depth_loss_dense = compute_dense_depth_error(point_cloud_lidar_dense, scene, depth_P, render_pkg_depth['depth'],
                                             depth_mask2.detach(), depth_ratio=opt.dense_ratio, temperature=opt.dense_temp)


        end_time_loss_depth = time.perf_counter()
        execution_time_loss_depth = end_time_loss_depth - start_time
        match_loss = 0.
        if scene.sensor_trajectory.rotation_cl_delta.requires_grad and current_scale_index<3:
            visualize_match = False
            match_loss = compute_match_loss(
                scene=scene,
                vind=vind,
                all_depths=all_depths,
                all_depth_masks = all_depth_masks,
                resolution_scale=resolution_scales[current_scale_index],
                gt_image=gt_image,
                gt_gradient_mask=gt_gradient_mask,
                gt_pixel_mask = gt_pixel_mask,
                rendered_depth=rendered_depth,
                depth_mask=depth_mask,
                window_size=match_loss_window_size,
                use_flow=True,
                visualize = visualize_match)


        end_time_loss_match = time.perf_counter()
        execution_time_loss_match = end_time_loss_match - start_time

        d_loss = depth_loss * 10 + depth_loss_dense * 10 * opt.dense_lr
        d_loss += match_loss

        p_loss = torch.zeros_like(d_loss)
        if (not scene.sensor_trajectory.rotation_cl_delta.requires_grad  ) or current_scale_index >=3:
            p_updated = True
            p_loss += ((1.0 - opt.lambda_dssim) * Ll1)
            p_loss += opt.lambda_dssim * (1.0 - ssim_value)


        # Scale norm loss
        observable_mask = radii > 0.
        visible_scaling = gaussians.get_scaling[observable_mask]
        scale_constraint = torch.max(visible_scaling, dim=1).values / (torch.min(visible_scaling, dim=1).values + 1e-7)
        clip_constraint = torch.clip(scale_constraint - 10, min=0.)
        scaling_loss = torch.mean(clip_constraint)
        d_loss += 3e-2 * scaling_loss

        #-add-
        if TENSORBOARD_FOUND:
            writer.add_scalar('Loss/total_ema', ema_loss_for_log, iteration)
            # photo_loss
            writer.add_scalar('Loss/total_loss', p_loss.item(), iteration)
            writer.add_scalar('Loss/l1_loss', Ll1.item(), iteration)
            writer.add_scalar('Loss/ssim_value', ssim_value.item(), iteration)
            match_loss_f = match_loss
            if type(match_loss_f) is torch.Tensor:
                match_loss_f = match_loss_f.item()

            writer.add_scalar('Loss/match_loss', match_loss_f, iteration)
            writer.add_scalar('Loss/depth_loss', depth_loss.item(), iteration)
            depth_loss_dense_f = depth_loss_dense
            if type(depth_loss_dense_f) is torch.Tensor:
                depth_loss_dense_f = depth_loss_dense_f.item()
            writer.add_scalar('Loss/depth_loss_dense', depth_loss_dense_f, iteration)

        if TENSORBOARD_FOUND and (iteration % 1000 == 0 or (iteration > 9000 and iteration % 300 == 0)) and dataset.save_log:
            writer.add_image('Images/rendered_image', image, iteration)
            writer.add_image('Images/ground_truth_image', gt_image, iteration)

            visualize_match = False
            origin_img, reporj_img = get_reproj_images(
                scene=scene,
                vind=vind,
                all_depths=all_depths,
                all_depth_masks = all_depth_masks,
                resolution_scale=resolution_scales[current_scale_index],
                gt_image=gt_image,
                gt_gradient_mask=gt_gradient_mask,
                gt_pixel_mask = gt_pixel_mask,
                rendered_depth=rendered_depth,
                depth_mask=depth_mask,
                window_size=match_loss_window_size,
                use_flow=True,
                visualize = visualize_match)
            
            writer.add_image('Images/reproj_origin_image', origin_img, iteration)
            writer.add_image('Images/reporj_img', reporj_img, iteration)


        rotation_err, translation_err = scene.sensor_trajectory.get_extrinsic_error()
        rotation_err_geodesic, _ = scene.sensor_trajectory.get_extrinsic_error_dh()
        if TENSORBOARD_FOUND:
            writer.add_scalar('Loss/rot_Error', torch.norm(rotation_err).item(), iteration)
            writer.add_scalar('Loss/t_Error', torch.norm(translation_err).item(), iteration)
            writer.add_scalar('Loss/rot_Error_geodesic', rotation_err_geodesic.item(), iteration)


        def is_trainable_loss(x):
            return isinstance(x, torch.Tensor) and x.requires_grad

        def safe_backward(loss, tag, retain_graph=True):
            if is_trainable_loss(loss):
                if not torch.isfinite(loss):
                    print(f"[skip backward] non-finite {tag}")
                    return
                loss.backward(retain_graph=retain_graph)
                had_bad_grad = sanitize_extrinsic_grads(scene, clip_norm=1.0)
                if had_bad_grad:
                    print(f"[NaN/Inf DETECTED] sanitized extrinsic gradients after {tag}")
                nan_report(
                    f"after {tag}",
                    rot_grad=scene.sensor_trajectory.rotation_cl_delta.grad,
                    trans_grad=scene.sensor_trajectory.translation_cl_delta.grad,
                )
            else:
                pass

        d_loss.backward(retain_graph=True)

        if p_updated:
            geom_params = [gaussians._xyz, gaussians._scaling, gaussians._rotation]
            handles = []

            for p in geom_params:
                if p.requires_grad:
                    h = p.register_hook(lambda g: torch.zeros_like(g) if g is not None else None)
                    handles.append(h)

            p_loss.backward()

            for h in handles:
                h.remove()


        iter_end.record()

        with torch.no_grad():
            p_loss_log = p_loss.item() if torch.isfinite(p_loss) else 0.0
            d_loss_log = d_loss.item() if torch.isfinite(d_loss) else 0.0
            ema_loss_for_log = 0.4 * (p_loss_log + d_loss_log) + 0.6 * ema_loss_for_log

            if iteration % 500 == 0:
                rotation_err, translation_err = scene.sensor_trajectory.get_extrinsic_error()

                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}"
                })

                tqdm.write(
                    f"[Iter {iteration}] "
                    f"rotation_err: {rotation_err.cpu().numpy().tolist()} | "
                    f"translation_err: {translation_err.cpu().numpy().tolist()}"
                )

                progress_bar.update(500)

            if iteration == opt.iterations:
                progress_bar.close()


            # Optimizer step
            if iteration < opt.iterations:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if iteration >= change_iteration[current_scale_index] + 1000:
                    scene.sensor_trajectory.start_optimize_rotation_cl()
                    scene.sensor_trajectory.start_optimize_translation_cl()

                    if iteration >= change_iteration[current_scale_index] + per_level_iterations[
                        current_scale_index] - 10:
                        scene.sensor_trajectory.stop_optimize_rotation_cl()
                        scene.sensor_trajectory.stop_optimize_translation_cl()

                    if iteration % 30 ==0 or (current_scale_index <= 2 and iteration %10 == 0):
                        did_pose_update = False
                        had_nonfinite_grad = sanitize_extrinsic_grads(scene, clip_norm=1.0)

                        rot_grad = scene.sensor_trajectory.rotation_cl_delta.grad
                        trans_grad = scene.sensor_trajectory.translation_cl_delta.grad
                        if had_nonfinite_grad or has_nan(rot_grad) or has_nan(trans_grad):
                            print(f"[Iter {iteration}] Skip camera step due to non-finite gradient.")
                            Nan_detected = 3
                            scene.sensor_trajectory.optimizer.zero_grad(set_to_none=True)
                        else:
                            scene.sensor_trajectory.optimizer.step()
                            normalize_quaternion_(scene.sensor_trajectory.rotation_cl_delta)

                            nan_report(
                                "after step",
                                rot_delta=scene.sensor_trajectory.rotation_cl_delta,
                                trans_delta=scene.sensor_trajectory.translation_cl_delta,
                            )

                            scene.sensor_trajectory.update_extrinsics()
                            normalize_quaternion_(scene.sensor_trajectory.rotation_cl)
                            did_pose_update = True

                            nan_report(
                                "after update_extrinsics",
                                rot=scene.sensor_trajectory.rotation_cl,
                                trans=scene.sensor_trajectory.translation_cl,
                            )

                            scene.sensor_trajectory.optimizer.zero_grad(set_to_none=True)

                        if did_pose_update:
                            all_depths, all_depth_masks = generate_all_depths(scene, pipe, background,
                                                                              resolution_scales[current_scale_index],
                                                                              train_test_exp=dataset.train_test_exp,
                                                                              full_cut_size=cut_full_size,
                                                                              min_depth=min_depth, max_depth=max_depth)
                else:
                    scene.sensor_trajectory.stop_optimize_rotation_cl()
                    scene.sensor_trajectory.stop_optimize_translation_cl()

    rotation_err, translation_err = scene.sensor_trajectory.get_extrinsic_error()
    rotation_err_geodesic, _ = scene.sensor_trajectory.get_extrinsic_error_dh()
    rot_mean_err = torch.norm(rotation_err).item()
    rot_mean_err_geodesic = rotation_err_geodesic.item()
    trans_mean_err = torch.norm(translation_err).item()
    
    print('Final rotation error:', rot_mean_err)
    print('Final translation error:', trans_mean_err)

    calibrated_rot = pytorch3d.transforms.quaternion_to_matrix(scene.sensor_trajectory.rotation_cl.detach().cpu()).numpy()
    calibrated_trans = scene.sensor_trajectory.translation_cl.detach().cpu().numpy()

    print('Calibrated rotation: ')
    print(calibrated_rot)
    print('Calibrated translation: ')
    print(calibrated_trans)

    # Save to extrinsic.json
    extrinsic_info = {
        "calibration": {
            "rotation": calibrated_rot.tolist(),
            "translation": calibrated_trans.tolist()
        },
        "errors": {
            "rotation": {
                "mean": rot_mean_err,
                "geodesic_mean": rot_mean_err_geodesic,
                "axis": rotation_err.detach().cpu().numpy().tolist()
            },
            "translation": {
                "mean": trans_mean_err,
                "axis": translation_err.detach().cpu().numpy().tolist()
            }
        }
    }

    with open(os.path.join(log_dir, "extrinsic.json"), "w") as f:
        json.dump(extrinsic_info, f, indent=4)

    # Save parameters
    with open(os.path.join(log_dir, "cfg_args.json"), 'w') as f:
        try:
            cfg = {
                "dataset": vars(dataset),
                "opt": vars(opt),
                "pipe": vars(pipe),
                "args": vars(args) if 'args' in locals() else {}
            }
            json.dump(cfg, f, indent=4)
        except Exception as e:
            print(f"Failed to save args.json: {e}")

    if TENSORBOARD_FOUND and writer is not None:
        writer.close()
        print("--- TensorBoard ED. ---")

    if dataset.save_ply:
        gaussians.save_ply(os.path.join(log_dir, "point_cloud.ply"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[10, 1_000, 5_000, 7_000, 10000, 15000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 5_000, 7_000, 10000, 15000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000, 5000, 10000, 15000, 20000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--cam_id", type=str, default=None)
    parser.add_argument("--data_seq", type=int, default=None)
    parser.add_argument("--dense_lr", type=float, default=1.0)
    parser.add_argument("--voxel_res", type=float, default=0.1)
    parser.add_argument("--dense_sharpness", type=float, default=100.0)
    parser.add_argument("--dense_ratio", type=float, default=0.05)
    parser.add_argument("--dense_stage", type=int, default=1)
    parser.add_argument("--no_save_ply", dest="save_ply", action="store_false")
    parser.add_argument("--no_save_log", dest="save_log", action="store_false")
    parser.set_defaults(save_ply=True, save_log=True)


    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    lp.cam_id = args.cam_id
    lp.data_seq = args.data_seq
    lp.seed = args.seed
    op.dense_lr = args.dense_lr
    op.voxel_res = args.voxel_res
    lp.save_ply = args.save_ply
    lp.save_log = args.save_log
    args.dense_temp = 1.0 / args.dense_sharpness
    op.dense_temp = args.dense_temp
    op.dense_ratio = args.dense_ratio
    op.dense_stage = args.dense_stage
    log_path = args.log_path
    seed = args.seed

    # Initialize system state (RNG)
    safe_state(args.quiet, seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, log_path)

    # All done
    print("\nTraining complete.")
