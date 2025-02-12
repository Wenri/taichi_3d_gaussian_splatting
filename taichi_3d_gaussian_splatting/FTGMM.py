#
# Authors: Bingchen Gong <gongbingchen@gmail.com>
#
from collections import UserList
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import pytorch3d.transforms
from tqdm import trange
import os

from .GaussianPointCloudScene import GaussianPointCloudScene

class BigList(UserList):
    def __repr__(self):
        # Only show basic info or partial data
        return f'<BigList len={len(self.list)}>'


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


def get_sample_coords(gmm, grid_size):
    device = gmm.component_distribution.mean.device

    # Get bounding box of GMM
    bbox_min, bbox_max = estimate_gmm_bbox(gmm)

    # Generate coordinate grid over the bbox range
    coords_x = torch.linspace(bbox_min[0], bbox_max[0], grid_size, device=device)
    coords_y = torch.linspace(bbox_min[1], bbox_max[1], grid_size, device=device)
    coords_z = torch.linspace(bbox_min[2], bbox_max[2], grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords_x, coords_y, coords_z, indexing='ij')

    # Stack to get coordinates tensor of shape (H,W,D,3)
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (H,W,D,3)
    return coords


def get_fourier_coords(grid_size, bbox_min, bbox_max, device: Optional[torch.device] = None):
    # Build the frequency grid corresponding to the DFT.
    # The spatial grid is defined on [bbox_min, bbox_max]. For each dimension, compute the corresponding frequency bins.
    L = bbox_max - bbox_min  # Extent in each dimension, shape: (D,)
    freq_grids = []
    for i in range(len(L)):
        # torch.fft.fftfreq returns frequencies in cycles per unit; multiply by 2π to obtain radians.
        freqs = torch.fft.fftfreq(grid_size, d=(L[i].item() / grid_size)) * 2 * torch.pi
        freq_grids.append(freqs.to(device))
    # Create a 3D meshgrid of frequency coordinates.
    kx, ky, kz = torch.meshgrid(freq_grids[0], freq_grids[1], freq_grids[2], indexing='ij')
    # Stack to form a tensor of shape (grid_size, grid_size, grid_size, 3)
    k_grid = torch.stack([kx, ky, kz], dim=-1)
    # Since volume_fft was produced via fftshift, we also shift the frequency grid.
    k_grid = torch.fft.fftshift(k_grid, dim=(0, 1, 2))

    return k_grid


def sample_gmm(gmm: MixtureSameFamily, grid_size: int = 96, chunk_size=1):
    device = gmm.component_distribution.mean.device
    coords = get_sample_coords(gmm, grid_size)

    # Compute GMM log probabilities in chunks to avoid OOM
    volume = torch.zeros((grid_size, grid_size, grid_size), device=device)

    with torch.no_grad():
        # Process ${chunk_size} slices at a time
        for i in trange(0, grid_size, chunk_size, desc="Computing GMM log probabilities in chunks"):
            end_idx = min(i + chunk_size, grid_size)
            coords_chunk = coords[i:end_idx]  # (chunk,W,D,3)
            volume[i:end_idx] = gmm.log_prob(coords_chunk)


    # Visualize center slices along each axis
    os.makedirs('vis', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    grid_size = volume.shape[0]
    center_idx = grid_size // 2

    # Plot center slices
    axes[0].imshow(volume[center_idx, :, :].detach().cpu().numpy())
    axes[0].set_title('YZ slice (center X)')

    axes[1].imshow(volume[:, center_idx, :].detach().cpu().numpy())
    axes[1].set_title('XZ slice (center Y)')

    axes[2].imshow(volume[:, :, center_idx].detach().cpu().numpy())
    axes[2].set_title('XY slice (center Z)')

    plt.tight_layout()
    plt.savefig('vis/grid_gt.png')
    plt.close()

    # Convert log probabilities to probabilities if needed
    volume = torch.exp(volume)
    return volume


def transform_volume_to_fourier(volume: torch.Tensor):
    """Transform a volume into Fourier space by applying FFT

    Args:
        volume: Input volume tensor of shape (grid_size, grid_size, grid_size)

    Returns:
        Fourier transform of volume as complex tensor of shape (grid_size, grid_size, grid_size)
    """

    # Normalize volume to sum to 1
    volume = volume / volume.sum()

    # Apply 3D FFT
    fourier_volume = torch.fft.fftn(volume)

    # Shift zero frequency component to center
    fourier_volume = torch.fft.fftshift(fourier_volume, dim=(0, 1, 2))

    # Visualize middle slice magnitude and phase spectrum
    os.makedirs('vis', exist_ok=True)
    grid_size = volume.shape[0]
    mid = grid_size // 2

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

    # Plot magnitude for each dimension
    mag_x = torch.abs(fourier_volume[mid, :, :].detach().cpu()).numpy()
    mag_y = torch.abs(fourier_volume[:, mid, :].detach().cpu()).numpy()
    mag_z = torch.abs(fourier_volume[:, :, mid].detach().cpu()).numpy()

    im1 = ax1.imshow(mag_x)
    ax1.set_title('Magnitude Spectrum (YZ plane)')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(mag_y)
    ax2.set_title('Magnitude Spectrum (XZ plane)')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(mag_z)
    ax3.set_title('Magnitude Spectrum (XY plane)')
    plt.colorbar(im3, ax=ax3)

    # Plot phase for each dimension
    phase_x = torch.angle(fourier_volume[mid, :, :].detach().cpu()).numpy()
    phase_y = torch.angle(fourier_volume[:, mid, :].detach().cpu()).numpy()
    phase_z = torch.angle(fourier_volume[:, :, mid].detach().cpu()).numpy()

    im4 = ax4.imshow(phase_x)
    ax4.set_title('Phase Spectrum (YZ plane)')
    plt.colorbar(im4, ax=ax4)

    im5 = ax5.imshow(phase_y)
    ax5.set_title('Phase Spectrum (XZ plane)')
    plt.colorbar(im5, ax=ax5)

    im6 = ax6.imshow(phase_z)
    ax6.set_title('Phase Spectrum (XY plane)')
    plt.colorbar(im6, ax=ax6)

    plt.tight_layout()
    plt.savefig('vis/volume_fourier_spectrum.png')
    plt.close()

    return fourier_volume


def transform_gmm(gmm: MixtureSameFamily):
    bbox_min, bbox_max = estimate_gmm_bbox(gmm)
    f1 = transform_gmm_to_fourier1(gmm, bbox_min, bbox_max)
    alpha, complex_means, fourier_cov = transform_gmm_to_fourier_params(gmm)

    k = torch.as_tensor([[1, 2, 3]], dtype=alpha.dtype, device=alpha.device)
    v1 = f1(k)
    # v4 = fourier_response_at_k(k, alpha, complex_means, fourier_cov)
    return v1


@torch.no_grad()
def compare_gmm_volume_to_transforms(gmm: MixtureSameFamily, volume: torch.Tensor):
    """
    Compare the discrete FFT of `volume` to the closed-form GMM Fourier transform
    computed at matching frequency indices.
    """
    device = volume.device
    grid_size = volume.shape[0]  # Assuming volume is (N,N,N)

    # 1) Compute the DFT of the volume:
    #    volume_fft: shape (N, N, N), complex
    volume_fft = transform_volume_to_fourier(volume)

    # 2) Estimate the bounding box used to create 'volume', so we can build f1 consistently.
    bbox_min, bbox_max = estimate_gmm_bbox(gmm)  # or use your known bounding box
    # (Optionally expand it slightly if you originally did so in sample_gmm)

    # 3) Build the GMM -> Fourier function that matches the DFT convention:
    f1 = transform_gmm_to_fourier1(gmm, bbox_min, bbox_max)

    freqs = get_fourier_coords(grid_size, bbox_min, bbox_max, device)

    # 5) Compute the analytic GMM Fourier transform f1(k) on all these indices:
    #    We'll flatten k_indices and process in batches to avoid OOM:
    freq_grid_flat = freqs.view(-1, 3)  # (grid_size^3, 3)
    batch_size = 500  # Process 100 points at a time
    f1_values_flat = BigList()

    for i in trange(0, freq_grid_flat.shape[0], batch_size, desc="Computing GMM Fourier transform"):
        batch = freq_grid_flat[i:i + batch_size]
        f1_values_batch = f1(batch)  # shape (batch_size,) complex
        f1_values_flat.append(f1_values_batch)

    # Concatenate batches
    f1_values_flat = torch.cat(f1_values_flat.data)
    # reshape back
    f1_volume = f1_values_flat.reshape(grid_size, grid_size, grid_size)

    # 6) Now compare magnitude (and/or phase) to volume_fft:
    #    For example, compute an L2 error or correlation.
    #    NOTE: volume_fft is shape (N,N,N) complex, same for f1_values.
    #    a) Magnitude difference
    diff = torch.abs(volume_fft - f1_volume)
    mag_err = diff.mean().item()

    #    b) Possibly measure correlation in magnitude or real part
    corr_num = torch.sum((volume_fft.real * f1_volume.real) + (volume_fft.imag * f1_volume.imag))
    corr_den = torch.sqrt(
        torch.sum(volume_fft.real ** 2 + volume_fft.imag ** 2) * torch.sum(f1_volume.real ** 2 + f1_volume.imag ** 2))
    corr = (corr_num / corr_den).item()

    print(f"Mean absolute difference in Fourier space: {mag_err:.6f}")
    print(f"Cosine similarity in Fourier space:        {corr:.6f}")

    # 7) You could also visualize slices:
    os.makedirs('vis', exist_ok=True)
    mid = grid_size // 2
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("DFT(volume) magnitude (central slice)")
    plt.imshow(torch.abs(volume_fft[:, :, mid]).cpu().numpy())
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title("Analytic GMM FT magnitude (central slice)")
    plt.imshow(torch.abs(f1_volume[:, :, mid]).cpu().numpy())
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title("DFT(volume) phase (central slice)")
    plt.imshow(torch.atan(torch.tan(torch.angle(volume_fft[:, :, mid]))).cpu().numpy())
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title("Analytic GMM FT phase (central slice)")
    plt.imshow(torch.atan(torch.tan(torch.angle(f1_volume[:, :, mid]))).cpu().numpy())
    plt.colorbar()

    plt.tight_layout()
    plt.savefig('vis/compare_volume_to_transforms.png')
    plt.close()

    return mag_err, corr


def transform_gmm_to_fourier1(gmm: MixtureSameFamily, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
    """
    Transform GMM into Fourier space using the closed-form Fourier transform of Gaussians.
    The returned function f1 takes as input frequency vectors k (with k given in cycles per unit,
    as produced by torch.fft.fftfreq) and returns the Fourier transform (both real and imaginary parts)
    that is directly comparable to the DFT computed from a volume.

    The conversion from cycles to angular frequency (radians) is performed and a global phase shift
    is applied to account for the fact that the spatial grid spans [bbox_min, bbox_max].

    Args:
        gmm: Gaussian Mixture Model (MixtureSameFamily)
        grid_size: Size of the grid (assumed cubic) used in volume sampling.
        bbox_min: Tensor of shape (D,) representing minimum coordinates of the domain.
        bbox_max: Tensor of shape (D,) representing maximum coordinates of the domain.

    Returns:
        A function f1(k) that evaluates the closed-form Fourier transform of the GMM at frequencies k.
    """
    # Extract parameters from the GMM.
    mixture_dist = gmm.mixture_distribution  # Categorical distribution; shape: (batch_shape, num_components)
    component_dist = gmm.component_distribution  # MultivariateNormal; shape: (batch_shape, num_components, D)
    weights = mixture_dist.probs  # (batch_shape, num_components)
    means = component_dist.mean  # (batch_shape, num_components, D)
    covariances = component_dist.covariance_matrix  # (batch_shape, num_components, D, D)
    event_shape = means.shape[-1]

    # Compute the center of the spatial grid.
    center = (bbox_min + bbox_max) / 2.0  # (D,)

    def fourier_gmm(k):
        # k is expected to have shape (..., D) in cycles per unit.
        k = torch.as_tensor(k, dtype=means.dtype, device=means.device)
        if k.shape[-1] != event_shape:
            raise ValueError(f"Last dimension of k must be {event_shape}")

        # We now adjust the means so that the phase is computed relative to the center
        # of the bounding box. This shift is essential for matching the DFT result.
        shifted_means = means - center  # (batch_shape, num_components, D)

        # --- Compute the first term: -i k^T (μ_i - center) ---
        # Expand k so that it can broadcast with (batch_shape, num_components, D).
        # Original k is shape (..., D). We unsqueeze two dims to obtain shape (..., 1, 1, D).
        k_expanded = k.unsqueeze(-2).unsqueeze(-2)  # shape: (..., 1, 1, D)
        ik_dot_mu = -1j * torch.sum(k_expanded * shifted_means, dim=-1)
        # Now, ik_dot_mu has shape (..., batch_shape, num_components)

        # --- Compute the quadratic term: -0.5 * k^T Σ_i k ---
        # Prepare k for the quadratic form: add an extra dim at the end.
        k_expanded_mat = k_expanded.unsqueeze(-1)  # shape: (..., 1, 1, D, 1)
        # Expand covariances to have leading dimensions matching k.
        # covariances has shape (batch_shape, num_components, D, D) and we add a leading dim.
        covariances_expanded = covariances.unsqueeze(0)  # shape: (1, batch_shape, num_components, D, D)
        # Expand k_expanded_mat to match the batch dimensions of covariances.
        batch_dims = k_expanded_mat.shape[:-3]  # any leading dimensions from k
        k_expanded_mat = k_expanded_mat.expand(*batch_dims, *covariances.shape[:-2], event_shape, 1)
        # Multiply: Σ_i * k
        sigma_k = torch.matmul(covariances_expanded, k_expanded_mat)  # shape: (..., batch_shape, num_components, D, 1)
        # Now compute k^T (Σ_i * k)
        k_expanded_transpose = k_expanded_mat.transpose(-2, -1)  # shape: (..., batch_shape, num_components, 1, D)
        k_sigma_k = torch.matmul(k_expanded_transpose, sigma_k)
        k_sigma_k = k_sigma_k.squeeze(-1).squeeze(-1)  # shape: (..., batch_shape, num_components)

        # Combine the two terms into the exponent.
        exponent = ik_dot_mu - 0.5 * k_sigma_k  # shape: (..., batch_shape, num_components)

        # --- Weighted sum over components ---
        # Expand weights so that they can broadcast with exponent.
        weights_expanded = weights.unsqueeze(0)  # shape: (1, batch_shape, num_components)
        weighted_exponentials = weights_expanded * torch.exp(exponent)  # shape: (..., batch_shape, num_components)
        fourier_transform = weighted_exponentials.sum(dim=-1)  # sum over the components

        return fourier_transform

    return fourier_gmm


def optimize_grid_gmm(gmm: MixtureSameFamily, grid_size: int = 36, chunk_size=9):
    device = gmm.component_distribution.mean.device

    # Get bounding box of GMM
    bbox_min, bbox_max = estimate_gmm_bbox(gmm)

    # Create the grid module for 3D
    mygrid = MyGrid(grid_size=grid_size).to(device)
    # Define optimizer
    optimizer = optim.Adam(mygrid.parameters(), lr=1e-1)
    # Number of epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        with torch.no_grad():
            # Reduce number of samples to prevent OOM
            coords_gmm = gmm.sample([1, 1, 10000])  # Sample from GMM (reduced from 40000)
            coords_rnd = torch.rand_like(coords_gmm) * (bbox_max - bbox_min) + bbox_min  # Random samples within bbox
            coords_flat = torch.cat([coords_gmm, coords_rnd], dim=1)

            # Compute true log_probs from GMM at the corresponding coordinates
            true_log_probs = gmm.log_prob(coords_flat)  # (H*W*D)

        # The grid's values are of shape (1,1,H,W,D), flatten to (H*W*D)
        grid_values = nn.functional.logsigmoid(mygrid(coords_flat))  # Predicted log_probs at grid positions

        # Compute loss
        loss = nn.functional.l1_loss(grid_values.squeeze(), true_log_probs.squeeze())

        # Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    # After training, visualize middle slice of volume
    os.makedirs('vis', exist_ok=True)
    fig = plt.figure()
    plt.imshow(torch.sigmoid(mygrid.grid.squeeze()[:, :, grid_size // 2].detach().cpu()).numpy())
    plt.savefig('vis/grid_displacement.png')
    plt.close()


def estimate_gmm_bbox(gmm: MixtureSameFamily, std_multiplier: float = 2.0):
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
    bbox_min = torch.empty(3, device=means.device)
    bbox_max = torch.empty(3, device=means.device)

    # Plot histograms of means for each dimension
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dimensions = ['X', 'Y', 'Z']
    for i in range(3):
        data = means[:, i].detach().cpu().numpy()

        # Fit normal distribution
        mu, std = norm.fit(data)

        # sel = np.abs(data - mu) < std
        # data = data[sel]
        # mu, std = norm.fit(data)

        # Plot histogram
        _, bins_edges, _ = axes[i].hist(data, bins=100, density=True, alpha=0.6)

        # Plot fitted PDF
        p = norm.pdf(bins_edges, mu, std)
        axes[i].plot(bins_edges, p, 'r', linewidth=2)

        # Set labels
        title = f'Distribution of {dimensions[i]} means\nFit: μ = {mu:.2f}, σ = {std:.2f}'
        axes[i].set_title(title)
        axes[i].set_xlabel(f'{dimensions[i]} coordinate')
        axes[i].set_ylabel('Density')

        bbox_min[i] = (mu - 3 * std).item()
        bbox_max[i] = (mu + 3 * std).item()

    plt.tight_layout()
    plt.savefig('vis/gmm_means_distribution.png')
    plt.close()

    return bbox_min, bbox_max


def ft_grab_scene(scene):
    gmm = define_gmm(scene=scene)
    volume = sample_gmm(gmm, grid_size=35)
    compare_gmm_volume_to_transforms(gmm, volume)
