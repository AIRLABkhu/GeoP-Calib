import torch

def find_nearest_gaussian(points, gaussians):
    """
    Find the index of the nearest Gaussian center for each point.

    Args:
        points (torch.Tensor): Input point cloud with shape [N, 3].
        gaussians (torch.Tensor): Gaussian center coordinates with shape [M, 3].

    Returns:
        torch.Tensor: Indices of the nearest Gaussian center for each point, shape [N].
    """
    distances = torch.cdist(points, gaussians)

    nearest_indices = torch.argmin(distances, dim=1)

    return nearest_indices

def batch_find_nearest(points, gaussians, batch_size=10000):
    nearest = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        dists = torch.cdist(batch, gaussians)
        nearest.append(torch.argmin(dists, dim=1))
    return torch.cat(nearest)
