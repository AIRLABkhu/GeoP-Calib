import numpy as np
import cv2
import matplotlib.pyplot as plt
import threading
from queue import Queue, Empty
from multiprocessing import Lock, Process
import torch.nn.functional as F
import torch


def project_lidar_to_image(points_lidar, image, K, T_lidar_to_cam, P = None):
    """
    Project LiDAR points to the image plane and visualize depth with color.

    Args:
        points_lidar: Nx3 NumPy array of points in the LiDAR frame.
        image: HxWx3 NumPy array in BGR format.
        K: 3x3 NumPy array of camera intrinsics.
        T_lidar_to_cam: 4x4 NumPy transform from LiDAR to camera frame.
        P: Optional projection matrix (unused in this function).

    Returns:
        HxWx3 NumPy array with projected points overlaid on the image.
    """
    N = points_lidar.shape[0]
    points_hom = np.hstack((points_lidar, np.ones((N, 1))))

    points_cam = (T_lidar_to_cam @ points_hom.T).T[:, :3]

    valid_z = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_z]

    if points_cam.shape[0] == 0:
        return image.copy()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]
    u = (fx * x / z + cx).astype(int)
    v = (fy * y / z + cy).astype(int)

    H, W = image.shape[:2]
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]

    if len(u) == 0:
        return image.copy()

    z_min, z_max = np.percentile(z, [2, 98])
    norm = plt.Normalize(z_min, z_max)
    cmap = plt.get_cmap('jet')

    colors = cmap(norm(z))[:, :3]
    colors = (colors * 255).astype(np.uint8)

    depth_buffer = np.full((H, W), np.inf)
    result = image.copy()

    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            result[v[i], u[i]] = colors[i]

    return result


def project_to_pixel_torch(points_3d, P, image_width, image_height):
    """Project 3D points in view space to image pixel coordinates (PyTorch)."""
    ones = torch.ones_like(points_3d[:,0:1])
    points_homo = torch.hstack([points_3d, ones])  # (N, 4)

    clip_coords = points_homo @ P.T

    w = clip_coords[:, 3].view(-1, 1)  # (N, 1)
    ndc_coords = clip_coords[:, :3] / w    # (N, 3)

    u = (ndc_coords[:, 0] + 1) * 0.5 * image_width
    v = (ndc_coords[:, 1] + 1) * 0.5 * image_height

    return torch.column_stack([u, v])

def project_to_pixel(points_3d, P, image_width, image_height):
    """Project 3D points in view space to image pixel coordinates (NumPy)."""
    ones = np.ones((points_3d.shape[0], 1), dtype=np.float32)
    points_homo = np.hstack([points_3d, ones])  # (N, 4)

    clip_coords = points_homo @ P.T

    w = clip_coords[:, 3].reshape(-1, 1)  # (N, 1)
    ndc_coords = clip_coords[:, :3] / w    # (N, 3)

    u = (ndc_coords[:, 0] + 1) * 0.5 * image_width
    v = (ndc_coords[:, 1] + 1) * 0.5 * image_height

    return np.column_stack([u, v])


def project_lidar_to_image_with_projection(points_lidar, image, P, T_lidar_to_cam):
    """
    Project LiDAR points to the image plane with a projection matrix.

    Args:
        points_lidar: Nx3 NumPy array of points in the LiDAR frame.
        image: HxWx3 NumPy array in BGR format.
        P: 4x4 projection matrix.
        T_lidar_to_cam: 4x4 NumPy transform from LiDAR to camera frame.

    Returns:
        HxWx3 NumPy array with projected points overlaid on the image.
    """
    N = points_lidar.shape[0]
    points_hom = np.hstack((points_lidar, np.ones((N, 1))))

    points_cam = (T_lidar_to_cam @ points_hom.T).T[:, :3]

    valid_z = points_cam[:, 2] > 0.1
    points_cam = points_cam[valid_z]

    if points_cam.shape[0] == 0:
        return image.copy()


    z = points_cam[:, 2]

    H, W = image.shape[:2]
    points_pixel = project_to_pixel(points_cam, P, W, H)

    u = points_pixel[:,0].astype(int)
    v = points_pixel[:,1].astype(int)

    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]

    if len(u) == 0:
        return image.copy()

    z_min, z_max = np.percentile(z, [2, 98])
    norm = plt.Normalize(z_min, z_max)
    cmap = plt.get_cmap('jet')

    colors = cmap(norm(z))[:, :3]
    colors = (colors * 255).astype(np.uint8)

    depth_buffer = np.full((H, W), np.inf)
    result = image.copy()

    for i in range(len(u)):
        if z[i] < depth_buffer[v[i], u[i]]:
            depth_buffer[v[i], u[i]] = z[i]
            result[v[i], u[i]] = colors[i]

    return result




class ImageViewer:
    def __init__(self, img = None):
        self.queue = Queue(maxsize=1)
        self.running = False
        self.window_name = "OpenCV Viewer"
        self.img = img
        self.mutex = Lock()

    def initialize_img(self, lidar_points, img, intrinsics, T_cl):
        self.img = project_lidar_to_image(lidar_points, img, intrinsics, T_cl)
        self.lidar_points = lidar_points
        self.intrinsics = intrinsics
        self.T_cl = T_cl
        self.start()

    def update_pose(self, T_cl):
        self.T_cl = T_cl
        self.mutex.acquire()
        self.img = project_lidar_to_image(self.lidar_points, self.img, self.intrinsics, self.T_cl)
        self.mutex.release()

    def _display_thread(self):
        """Internal display thread function."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while self.running:
            self.mutex.acquire()
            cv2.imshow(self.window_name, self.img)
            key = cv2.waitKey(30) & 0xFF
            self.mutex.release()
            if key == 27:
                self.stop()
        cv2.destroyAllWindows()

    def start(self):
        """Start the display thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._display_thread)
            self.thread.daemon = True
            self.thread.start()

    def update(self, image):
        """Update the displayed image (thread-safe)."""
        try:
            self.mutex.acquire()
            self.img = image
            self.mutex.release()
        except:
            pass

    def stop(self):
        """Stop the display thread."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)



def unproject_pixels_to_3d_torch(pixel_uv, depth_values, P, image_width, image_height):
    """
    Unprojects 2D pixel coordinates with depth back to 3D points in view space.
    This is the inverse of project_to_pixel_torch, assuming depth_values are Z_view.

    Args:
        pixel_uv (torch.Tensor): (N, 2) tensor of (u, v) pixel coordinates.
        depth_values (torch.Tensor): (N,) or (N,1) tensor of depth values (Z_view) for each pixel.
                                     These are the Z coordinates in the view/camera space.
        P (torch.Tensor): (4, 4) projection matrix used in the forward projection.
        image_width (float): Width of the image in pixels.
        image_height (float): Height of the image in pixels.

    Returns:
        torch.Tensor: (N, 3) tensor of 3D points (X_view, Y_view, Z_view) in view space.
                      Returns empty tensor if input pixel_uv is empty.
    """
    if pixel_uv.shape[0] == 0:
        return torch.empty((0, 3), device=pixel_uv.device, dtype=pixel_uv.dtype)

    u = pixel_uv[:, 0]
    v = pixel_uv[:, 1]
    Zv = depth_values.squeeze(-1) if depth_values.ndim == 2 else depth_values  # Ensure (N,)

    # Inverse viewport transform: Pixel to NDC
    # u = (x_ndc + 1) * 0.5 * image_width  => x_ndc = (2u / image_width) - 1
    # v = (y_ndc + 1) * 0.5 * image_height => y_ndc = (2v / image_height) - 1
    # This matches the forward function's NDC to pixel mapping.
    x_ndc = (2.0 * u / image_width) - 1.0
    y_ndc = (2.0 * v / image_height) - 1.0

    # We need to solve for Xv, Yv. We know Zv.
    # The forward projection equations are:
    # x_ndc * Wc = P[0,0]Xv + P[0,1]Yv + P[0,2]Zv + P[0,3]  (1)
    # y_ndc * Wc = P[1,0]Xv + P[1,1]Yv + P[1,2]Zv + P[1,3]  (2)
    # Wc         = P[3,0]Xv + P[3,1]Yv + P[3,2]Zv + P[3,3]  (3)
    #
    # This is a system of 3 linear equations for Xv, Yv, Wc for each point:
    # P[0,0]Xv + P[0,1]Yv - x_ndc*Wc = - (P[0,2]Zv + P[0,3])
    # P[1,0]Xv + P[1,1]Yv - y_ndc*Wc = - (P[1,2]Zv + P[1,3])
    # P[3,0]Xv + P[3,1]Yv - Wc       = - (P[3,2]Zv + P[3,3])
    #
    # Let sol = [Xv, Yv, Wc]^T.
    # M @ sol = B
    # M is (N, 3, 3), B is (N, 3, 1)

    num_points = pixel_uv.shape[0]
    M_batch = torch.zeros((num_points, 3, 3), device=P.device, dtype=P.dtype)
    B_batch = torch.zeros((num_points, 3, 1), device=P.device, dtype=P.dtype)

    # Construct M matrix for each point
    M_batch[:, 0, 0] = P[0, 0]
    M_batch[:, 0, 1] = P[0, 1]
    M_batch[:, 0, 2] = -x_ndc

    M_batch[:, 1, 0] = P[1, 0]
    M_batch[:, 1, 1] = P[1, 1]
    M_batch[:, 1, 2] = -y_ndc

    M_batch[:, 2, 0] = P[3, 0]
    M_batch[:, 2, 1] = P[3, 1]
    M_batch[:, 2, 2] = -1.0

    # Construct B vector for each point
    B_batch[:, 0, 0] = -(P[0, 2] * Zv + P[0, 3])
    B_batch[:, 1, 0] = -(P[1, 2] * Zv + P[1, 3])
    B_batch[:, 2, 0] = -(P[3, 2] * Zv + P[3, 3])

    # Solve the batched linear system
    # Some matrices in M_batch might be singular if P or ndc coords are weird.
    # e.g. if P[3,0]=P[3,1]=0 and -1=0 (problematic) or if leading 2x2 of P is singular.
    # Using try-catch or checking determinant could be added for robustness.
    try:
        solution_batch = torch.linalg.solve(M_batch, B_batch)  # Result is (N, 3, 1)
    except torch.linalg.LinAlgError as e:
        # Handle cases where matrix is singular for some points
        # For now, re-raise or return NaNs.
        # A more robust solution might involve SVD or pseudo-inverse,
        # or identifying problematic points.
        print(f"Linear algebra error during unprojection: {e}")
        # Create a NaN tensor of the expected shape
        nan_points_3d = torch.full((num_points, 3), float('nan'), device=P.device, dtype=P.dtype)
        return nan_points_3d

    Xv_batch = solution_batch[:, 0, 0]
    Yv_batch = solution_batch[:, 1, 0]
    # Wc_batch = solution_batch[:, 2, 0] # Wc is also solved for, can be used for sanity checks

    # The 3D point is (Xv, Yv, Zv)
    points_3d_unprojected = torch.stack([Xv_batch, Yv_batch, Zv], dim=-1)

    return points_3d_unprojected


# ----- Helper function to use the unprojection with a mask and depth_map -----
def get_3d_points_from_pixels_depth_mask_torch(
        depth_map,
        mask,
        P,
        image_width,
        image_height
):
    """
    Extracts 3D points from a depth map using a mask and unprojects them.

    Args:
        depth_map (torch.Tensor): (H, W) tensor of depth values (Z_view).
        mask (torch.Tensor): (H, W) boolean tensor, True for pixels to unproject.
        P (torch.Tensor): (4, 4) projection matrix.
        image_width (float): Width of the image.
        image_height (float): Height of the image.

    Returns:
        torch.Tensor: (N_masked, 3) tensor of 3D points in view space,
                      where N_masked is the number of True values in the mask.
                      Returns empty tensor if no pixels in mask.
    """
    if not torch.any(mask):
        return torch.empty((0, 3), device=depth_map.device, dtype=depth_map.dtype)

    # Get pixel coordinates (v, u) where mask is True
    v_coords, u_coords = torch.where(mask)  # v_coords are row indices, u_coords are col indices

    # Select depth values for these pixels
    depth_values_masked = depth_map[v_coords, u_coords]  # (N_masked,)

    # Stack u, v coordinates
    # Note: u_coords correspond to x-direction (width), v_coords to y-direction (height)
    pixel_uv_masked = torch.stack([u_coords.float(), v_coords.float()], dim=-1)  # (N_masked, 2)

    # Unproject
    points_3d_view_space = unproject_pixels_to_3d_torch(
        pixel_uv_masked,
        depth_values_masked,  # Will be (N_masked,)
        P,
        image_width,
        image_height
    )
    return points_3d_view_space


if __name__ == "__main__":
    H, W = 480, 640
    dummy_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_image = dummy_image * 0 + 122

    x = np.random.uniform(-5, 5, 10000)
    y = np.random.uniform(-5, 5, 10000)
    z = np.random.uniform(2, 10, 10000)
    points_lidar = np.vstack([x, y, z]).T

    K = np.array([[500, 0, 320],
                  [0, 500, 240],
                  [0, 0, 1]])

    T_lidar_to_cam = np.eye(4)
    T_lidar_to_cam[2, 3] = 1.0

    vis_image = project_lidar_to_image(points_lidar, dummy_image, K, T_lidar_to_cam)

    cv2.imshow('Projection Result', vis_image[..., ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_gradient_mask(image_tensor, ratio = 0.5):
    """
    Create a binary mask from per-pixel gradient magnitude using a mean threshold.

    Args:
        image_tensor (torch.Tensor): Input image tensor with shape [C, H, W].
                                    Values are expected to be in [0, 1] or normalized.

    Returns:
        torch.Tensor: Binary mask of shape [H, W].
    """
    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()

    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image_tensor.device)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image_tensor.device)

    sobel_kernel_x = sobel_kernel_x.repeat(image_tensor.shape[0], 1, 1, 1)
    sobel_kernel_y = sobel_kernel_y.repeat(image_tensor.shape[0], 1, 1, 1)

    gx = F.conv2d(image_tensor.unsqueeze(0), sobel_kernel_x, padding=1, groups=image_tensor.shape[0]).squeeze(0)
    gy = F.conv2d(image_tensor.unsqueeze(0), sobel_kernel_y, padding=1, groups=image_tensor.shape[0]).squeeze(0)

    gradient_magnitude = torch.sqrt(gx.pow(2) + gy.pow(2)).mean(dim=0)

    image_mean = torch.mean(image_tensor)

    mask = (gradient_magnitude > image_mean * ratio).float()

    return mask
