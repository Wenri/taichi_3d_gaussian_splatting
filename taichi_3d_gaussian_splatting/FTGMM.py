#
# Toy example of training a neural grid to sample from a GMM.
#
# Authors: Bingchen Gong <gongbingchen@gmail.com>
#
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import pytorch3d.transforms

from .GaussianPointCloudScene import GaussianPointCloudScene


def quaternions_to_scale_tril(quaternions, log_scales,
                              fallback_type: Optional[torch.dtype] = torch.float64,
                              fallback_scale=1e-5):
    """Quaternions to scale_tril

    Args:
        quaternions: (N, 4) in xyzw format
        log_scales: (N, 3)
        fallback_type: default to torch.float64
        fallback_scale: default to 1e-5

    Returns:
        scale_tril: (N, 3, 3)
    """
    # pytorch3d expects wxyz format, so we need to roll the quaternions to wxyz format
    quaternions = torch.roll(quaternions, 1, dims=-1)

    # Convert quaternions to rotation matrices (N, 3, 3)
    R = pytorch3d.transforms.quaternion_to_matrix(quaternions)

    # Convert log scales to actual scales
    scales = torch.exp(log_scales)  # (N, 3)

    # Construct covariance matrices for each Gaussian
    # Σ = R * S * S^T * R^T where S is diagonal scale matrix
    S = R * scales.unsqueeze(-2)  # (N, 3, 3)
    covariances = torch.bmm(S, S.transpose(-2, -1))  # (N, 3, 3)

    if fallback_type or fallback_scale:
        scale_tril, info = torch.linalg.cholesky_ex(covariances)
        if not torch.any(info):
            return scale_tril

        invalid = info > 0

        if fallback_type:
            valid = torch.logical_not(info)
            scale_tril = torch.empty_like(scale_tril)
            scale_tril[valid] = torch.linalg.cholesky(covariances[valid])

            quaternions = quaternions[invalid].to(dtype=fallback_type)
            log_scales = log_scales[invalid].to(dtype=fallback_type)
            scale_tril[invalid] = quaternions_to_scale_tril(
                quaternions, log_scales, fallback_type=None, fallback_scale=fallback_scale).to(scale_tril)
            
            return scale_tril

        if fallback_scale:
            scales = torch.clamp(scales[invalid], min=fallback_scale)
            # Construct covariance matrices for each Gaussian
            # Σ = R * S * S^T * R^T where S is diagonal scale matrix
            S = R[invalid] * scales.unsqueeze(-2)  # (N, 3, 3)
            covariances[invalid] = torch.bmm(S, S.transpose(-2, -1))  # (N, 3, 3)

    return torch.cholesky(covariances)


def define_gmm(batch_size=100, n_dim=2, scene: GaussianPointCloudScene = None) -> MixtureSameFamily:
    """Define a Gaussian Mixture Model either from parameters or a GaussianPointCloudScene
    
    Args:
        batch_size: Number of components if creating from parameters
        n_dim: Number of dimensions if creating from parameters  
        scene: Optional GaussianPointCloudScene to convert to GMM
        
    Returns:
        MixtureSameFamily: A PyTorch mixture distribution of the Gaussians
    """
    if scene is not None:
        # Only use valid points
        valid_mask = torch.logical_not(scene.point_invalid_mask)
        positions = scene.point_cloud[valid_mask]  # (N, 3)

        # Get rotation matrices and scales
        features = scene.point_cloud_features[valid_mask]
        scale_tril = quaternions_to_scale_tril(features[:, :4], features[:, 4:7])

        # Create multivariate normal with computed means and covariances
        comp = MultivariateNormal(positions, scale_tril=scale_tril, validate_args=False)

        # Get alpha values from features and activate them
        alphas = torch.sigmoid(features[:, 7])  # Sigmoid activation
        mix = Categorical(alphas)

    else:
        mu = (torch.rand(batch_size, n_dim) - 0.5) * 2
        cov = torch.eye(n_dim)[None].expand(batch_size, -1, -1) * 0.005

        comp = MultivariateNormal(mu, cov)
        mix = Categorical(torch.ones(batch_size, ))

    # Return mixture model
    return MixtureSameFamily(mix, comp)


class MyGrid(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.grid = nn.Parameter(torch.zeros(1, 1, grid_size, grid_size))

    def forward(self, x):
        return nn.functional.grid_sample(self.grid, x)


def train_sample_gmm():
    gmm = define_gmm()
    x = gmm.sample()

    grid_size = 800

    # Generate coordinate grid over the range [-1, 1]
    x_coords = y_coords = torch.linspace(-1, 1, grid_size)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')  # grid_x and grid_y of shape (H,W)

    # Stack to get coordinates tensor of shape (H,W,2)
    coords = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)

    # Compute GMM log probabilities at each coordinate to create an image
    with torch.no_grad():
        image = gmm.log_prob(coords)

    fig = plt.figure()
    plt.imshow(torch.exp(image.detach()).numpy())
    plt.savefig('grid_gt.png')
    plt.close()

    # Create the grid module
    mygrid = MyGrid(grid_size=grid_size)
    # Define optimizer
    optimizer = optim.Adam(mygrid.parameters(), lr=1e-1)
    # Number of epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        with torch.no_grad():
            coords_gmm = gmm.sample([1, 1, 40000])
            coords_rnd = torch.rand_like(coords_gmm) * 2 - 1
            coords_flat = torch.cat([coords_gmm, coords_rnd], dim=1)

            # Compute true log_probs from GMM at the corresponding coordinates
            true_log_probs = gmm.log_prob(coords_flat)  # (H*W)

        # The grid's values are of shape (1,1,H,W), flatten to (H*W)
        grid_values = nn.functional.logsigmoid(mygrid(coords_flat))  # Predicted log_probs at grid positions

        # Compute loss
        loss = nn.functional.l1_loss(grid_values.squeeze(), true_log_probs.squeeze())

        # Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    # After training, the grid parameter stores the GMM's log_probs at the grid coordinates
    fig = plt.figure()
    plt.imshow(torch.sigmoid(mygrid.grid.squeeze().detach()).numpy())
    plt.savefig('grid_displacement.png')
    plt.close()


def estimate_gmm_bbox(gmm: MixtureSameFamily, std_multiplier: float = 3.0):
    """Estimate bounding box of a GMM by considering means and covariances
    
    Args:
        gmm: Mixture of multivariate normals
        std_multiplier: How many standard deviations to include (default 3.0 covers ~99.7%)
    
    Returns:
        tuple: (min_coords, max_coords) tensors of shape (n_dim,)
    """
    means = gmm.component_distribution.mean  # (N, D)
    stds = gmm.component_distribution.stddev  # (N, D)

    # Calculate min/max by considering means +/- multiple of standard deviation
    mins = means - std_multiplier * stds  # (N, D)
    maxs = means + std_multiplier * stds  # (N, D)

    # Take min/max across all components
    bbox_min = torch.min(mins, dim=0)[0]  # (D,)
    bbox_max = torch.max(maxs, dim=0)[0]  # (D,)

    return bbox_min, bbox_max


if __name__ == '__main__':
    train_sample_gmm()
