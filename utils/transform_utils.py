import torch
import numpy as np


def project_points_to_depth(camera_pose, intrinsics, points_3d, image_shape):
    """
    Project a 3D point cloud into the camera frame and generate a depth map.

    Args:
    - camera_pose : 4x4 camera extrinsic matrix [[R, t], [0, 1]] that maps
                    world coordinates to camera coordinates.
    - intrinsics  : 3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    - points_3d   : Nx3 3D points in world coordinates.
    - image_shape : Output depth map size as (height, width).

    Returns:
    - depth_map : Depth map with non-projected regions filled with a small value.
    - mask      : Boolean-like mask marking valid projected regions.
    """
    H, W = image_shape

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    points_hom = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    points_cam = (camera_pose @ points_hom.T).T[:, :3]

    valid = points_cam[:, 2] > 0
    points_cam = points_cam[valid]
    if points_cam.size == 0:
        return np.zeros((H, W)), np.zeros((H, W), dtype=bool)

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = points_cam[:, 2]
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z

    u = np.round(fx * x + cx).astype(int)
    v = np.round(fy * y + cy).astype(int)

    valid_pixels = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[valid_pixels], v[valid_pixels], z[valid_pixels]
    if z.size == 0:
        return np.zeros((H, W)), np.zeros((H, W), dtype=bool)

    depth_map = np.full((H, W), np.inf)
    mask = np.zeros((H, W), dtype=np.float32)

    np.minimum.at(depth_map, (v, u), z)

    mask[v, u] = 1.
    depth_map[mask != 1.] = 1e-7

    return depth_map, mask
