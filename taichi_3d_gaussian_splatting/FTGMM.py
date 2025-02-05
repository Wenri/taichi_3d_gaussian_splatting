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


def get_fourier_coords(gmm, grid_size):
    device = gmm.component_distribution.mean.device

    # Get bounding box of GMM
    bbox_min, bbox_max = estimate_gmm_bbox(gmm)

    # Calculate frequency range based on bbox size
    # Use reciprocal of bbox size to determine frequency spacing
    freq_spacing = 2 * torch.pi / (bbox_max - bbox_min)

    # Generate frequency coordinates centered at 0
    # Grid spans [-grid_size/2, grid_size/2) in frequency space
    freq_x = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size, device=device) * freq_spacing[0]
    freq_y = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size, device=device) * freq_spacing[1]
    freq_z = torch.linspace(-grid_size // 2, grid_size // 2 - 1, grid_size, device=device) * freq_spacing[2]

    # Create frequency coordinate grid
    grid_kx, grid_ky, grid_kz = torch.meshgrid(freq_x, freq_y, freq_z, indexing='ij')

    # Stack to get wavevector coordinates tensor of shape (H,W,D,3)
    coords = torch.stack([grid_kx, grid_ky, grid_kz], dim=-1)

    return coords


def sample_gmm(gmm: MixtureSameFamily, grid_size: int = 36, chunk_size=9):
    device = gmm.component_distribution.mean.device
    coords = get_sample_coords(gmm, grid_size)

    # Compute GMM log probabilities in chunks to avoid OOM
    volume = torch.zeros((grid_size, grid_size, grid_size), device=device)

    with torch.no_grad():
        # Process ${chunk_size} slices at a time
        for i in range(0, grid_size, chunk_size):
            end_idx = min(i + chunk_size, grid_size)
            coords_chunk = coords[i:end_idx]  # (chunk,W,D,3)
            volume[i:end_idx] = gmm.log_prob(coords_chunk)

    # Visualize middle slice of volume
    fig = plt.figure()
    plt.imshow(torch.exp(volume[:, :, grid_size // 2].detach().cpu()).numpy())
    plt.savefig('grid_gt.png')
    plt.close()

    return volume


def transform_volume_to_fourier(volume: torch.Tensor):
    """Transform a volume into Fourier space by applying FFT
    
    Args:
        volume: Input volume tensor of shape (grid_size, grid_size, grid_size)
        
    Returns:
        Fourier transform of volume as complex tensor of shape (grid_size, grid_size, grid_size)
    """
    # Convert log probabilities to probabilities if needed
    if torch.min(volume) < 0:
        volume = torch.exp(volume)

    # Normalize volume to sum to 1
    volume = volume / volume.sum()

    # Apply 3D FFT
    fourier_volume = torch.fft.fftn(volume)

    # Shift zero frequency component to center
    fourier_volume = torch.fft.fftshift(fourier_volume)

    # Visualize middle slice magnitude spectrum
    grid_size = volume.shape[0]
    fig = plt.figure()
    plt.imshow(torch.abs(fourier_volume[:, :, grid_size // 2].detach().cpu()).numpy())
    plt.colorbar()
    plt.savefig('volume_fourier_spectrum.png')
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

    # 4) Build the frequency grid.
    # Compute the spacing in each dimension.
    # (Note: if get_sample_coords used torch.linspace(bbox_min, bbox_max, grid_size), the spacing is approximately:)
    d_x = (bbox_max[0] - bbox_min[0]) / grid_size
    d_y = (bbox_max[1] - bbox_min[1]) / grid_size
    d_z = (bbox_max[2] - bbox_min[2]) / grid_size
    freqs_x = torch.fft.fftfreq(grid_size, d=d_x, device=device)
    freqs_y = torch.fft.fftfreq(grid_size, d=d_y, device=device)
    freqs_z = torch.fft.fftfreq(grid_size, d=d_z, device=device)
    grid_freq_x, grid_freq_y, grid_freq_z = torch.meshgrid(freqs_x, freqs_y, freqs_z, indexing='ij')
    # Stack to get shape (grid_size, grid_size, grid_size, 3)
    freqs = torch.stack([grid_freq_x, grid_freq_y, grid_freq_z], dim=-1)
    freqs = torch.fft.fftshift(freqs)

    # 5) Compute the analytic GMM Fourier transform f1(k) on all these indices:
    #    We'll flatten k_indices and process in batches to avoid OOM:
    freq_grid_flat = freqs.view(-1, 3)  # (grid_size^3, 3)
    batch_size = 1000  # Process 1k points at a time
    f1_values_list = []

    for i in range(0, freq_grid_flat.shape[0], batch_size):
        batch = freq_grid_flat[i:i + batch_size]
        f1_values_batch = f1(batch)  # shape (batch_size,) complex
        f1_values_list.append(f1_values_batch)

    # Concatenate batches
    f1_values_flat = torch.cat(f1_values_list)
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
    import matplotlib.pyplot as plt
    mid = grid_size // 2

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("DFT(volume) magnitude (central slice)")
    plt.imshow(torch.abs(volume_fft[:, :, mid]).cpu().numpy())
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Analytic GMM FT magnitude (central slice)")
    plt.imshow(torch.abs(f1_volume[:, :, mid]).cpu().numpy())
    plt.colorbar()

    plt.tight_layout()
    plt.show()

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
    # Extract mixture parameters.
    weights = gmm.mixture_distribution.probs  # shape: (N,)
    means = gmm.component_distribution.mean  # shape: (N, D)
    covariances = gmm.component_distribution.covariance_matrix  # shape: (N, D, D)
    event_shape = means.shape[-1]  # D

    def fourier_gmm(k):
        # k is expected to have shape (..., D) in cycles per unit.
        k = torch.as_tensor(k, dtype=means.dtype, device=means.device)
        if k.shape[-1] != event_shape:
            raise ValueError(f"Last dimension of k must be {event_shape}")

        # Convert from cycles to angular frequency (radians per unit)
        k_ang = 2 * torch.pi * k  # now in radians

        # Compute the dot product for each component:
        # Expand k_ang to shape (..., 1, D) so it can broadcast with means (N, D).
        k_expanded = k_ang.unsqueeze(-2).unsqueeze(-2)  # shape: (..., 1, D)
        # Compute -i * k_ang dot μ for each component; result shape: (..., N)
        ik_dot_mu = -1j * torch.matmul(k_expanded, means.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        # Now compute the quadratic term for the covariance:
        # Expand k_ang to shape (..., 1, D, 1)
        k_expanded_cov = k_ang.unsqueeze(-2).unsqueeze(-1)  # shape: (..., 1, D, 1)
        # Expand covariances to shape (1, N, D, D) so that they broadcast.
        cov_expanded = covariances.unsqueeze(0)  # shape: (1, N, D, D)
        sigma_k = torch.matmul(cov_expanded, k_expanded_cov)  # shape: (..., N, D, 1)
        k_sigma_k = torch.matmul(k_expanded_cov.transpose(-2, -1), sigma_k).squeeze(-1).squeeze(-1)  # shape: (..., N)

        exponent = ik_dot_mu - 0.5 * k_sigma_k

        # Weighted sum over the mixture components.
        # Expand weights (shape (N,)) to broadcast with exponent (shape (..., N)).
        weights_expanded = weights.unsqueeze(0)  # shape: (1, N)
        fourier_vals = (weights_expanded * torch.exp(exponent)).sum(dim=-1)  # shape: (...,)

        # Apply a global phase shift to account for the spatial grid offset.
        # The sampled volume was generated over [bbox_min, bbox_max]. Using the center of the domain:
        center = (bbox_min + bbox_max) / 2  # shape: (D,)
        # Compute the phase shift exp(-i * k_ang dot center).
        phase_shift = torch.exp(-1j * torch.matmul(k_ang, center))

        return fourier_vals * phase_shift

    return fourier_gmm


def transform_gmm_to_fourier_params(gmm: MixtureSameFamily):
    r"""
    Transform a Gaussian Mixture Model (GMM) into its closed-form Fourier
    transform.  For each component:

        N(x; μ, Σ)   --->   exp[- i k^T μ - ½ k^T Σ k ]

    A GMM is a weighted sum of such Gaussians, so its transform is simply
    the weighted sum of these exponentials.  One can also rewrite each
    transform term as a "complex Gaussian" with mean -i Σ⁻¹ μ, same
    covariance Σ, and an amplitude factor exp(½ μ^T Σ⁻¹ μ).

    Args:
        gmm (MixtureSameFamily):
            A mixture model whose `component_distribution` is a batch of
            MultivariateNormal. That is,
              gmm = MixtureSameFamily(
                  mixture_distribution=Categorical(...),
                  component_distribution=MultivariateNormal(loc=..., covariance_matrix=...)
              )

    Returns:
        (new_weights, new_means, new_covs)
        Where:
          - new_weights (Tensor): shape (num_components,)
          - new_means   (Tensor): shape (num_components, D) [complex]
          - new_covs    (Tensor): shape (num_components, D, D) [real]

        These define a "complex" mixture in frequency space:
            F(k) = Σ_k [ new_weights_k * exp( -½ (k - new_means_k)ᵀ new_covs_k (k - new_means_k) ) ]
        but with new_means_k = -i Σ⁻¹ μ_k and an extra amplitude factor
        folded into new_weights_k = w_k * exp(½ μᵀ Σ⁻¹ μ).
    """
    assert isinstance(gmm.component_distribution, MultivariateNormal), (
        "transform_gmm_to_fourier currently only supports a MixtureSameFamily "
        "whose component_distribution is MultivariateNormal."
    )

    # Mixture weights: shape (num_components,)
    weights = gmm.mixture_distribution.probs

    # Means: shape (num_components, D)
    means = gmm.component_distribution.loc

    # Covariance matrices: shape (num_components, D, D)
    covs = gmm.component_distribution.covariance_matrix

    # Invert each covariance (batch-inverse for shape (num_components, D, D))
    covs_inv = torch.inverse(covs)

    # Compute the quadratic form μ^T Σ⁻¹ μ for each component
    # quad_k = means_k^T Σ_k⁻¹ means_k, shape (num_components,)
    quad_form = torch.einsum('bi,bij,bj->b', means, covs_inv, means)

    # new weight = original weight * exp( ½ μ^T Σ⁻¹ μ )
    new_weights = weights * torch.exp(0.5 * quad_form)

    # new mean = -i Σ⁻¹ μ (will be a complex Tensor)
    # shape: (num_components, D)
    # We use 1j in PyTorch for the imaginary unit.
    new_means = -1j * torch.einsum('bij,bj->bi', covs_inv, means)

    # new covariance = same Σ, but in "complex" sense
    new_covs = covs.clone()

    return new_weights, new_means, new_covs


def fourier_response_at_k(
        k: torch.Tensor,
        alpha: torch.Tensor,  # (K,) real
        means_complex: torch.Tensor,  # (K, D) complex
        covs: torch.Tensor  # (K, D, D) real
) -> torch.Tensor:
    """
    Evaluate the GMM's Fourier transform at frequency vector(s) k,
    given the "complex-Gaussian" parameters returned by
    transform_gmm_to_fourier_params.

    Args:
        k: shape (D,) or (N, D) real frequency vector(s)
        alpha: shape (K,) real amplitudes alpha_k
        means_complex: shape (K, D) complex means m_k
        covs: shape (K, D, D) real covariance matrices

    Returns:
        A complex tensor of shape () if k is (D,) or shape (N,) if k is (N,D).
        Specifically: sum_{k=1..K} alpha_k * exp( -1/2 (k - m_k)^T Sigma_k (k - m_k) ).
    """
    # Ensure k has a batch dimension for uniform processing:
    if k.ndim == 1:
        k = k.unsqueeze(0)  # shape: (1, D)
    # Now k is (N, D).

    K = alpha.shape[0]
    N = k.shape[0]
    D = k.shape[1]

    # Expand k to shape (K, N, D) so we can vectorize the exponent over components
    k_expanded = k.unsqueeze(0).expand(K, N, D)  # (K, N, D)
    m_expanded = means_complex.unsqueeze(1).expand(-1, N, -1)  # (K, N, D)

    # Difference: (k - m_k), shape: (K, N, D), complex
    diff = k_expanded - m_expanded  # still complex if means_complex is complex

    # We want the quadratic form: diff^T Sigma_k diff
    # So do a batched multiplication.  We can do it in steps:
    #   1) shape (K, N, 1, D) x shape (K, 1, D, D) => (K, N, 1, D)
    #   2) then dot with diff again => shape (K, N).

    diff_2 = diff.unsqueeze(-2)  # (K, N, 1, D)
    # Expand covs to (K, N, D, D) if needed
    covs_expanded = covs.unsqueeze(1).expand(-1, N, -1, -1)  # (K, N, D, D)

    quad_part = torch.matmul(diff_2, covs_expanded)  # (K, N, 1, D)
    # Now multiply by diff (K, N, D, 1)
    diff_3 = diff.unsqueeze(-1)  # (K, N, D, 1)
    exponent_terms = torch.matmul(quad_part, diff_3)  # (K, N, 1, 1)
    exponent_terms = exponent_terms.squeeze(-1).squeeze(-1)  # (K, N), complex

    # The exponent is -1/2 * exponent_terms
    # => shape (K, N) complex
    exponents = -0.5 * exponent_terms

    # exponentiate: exp(exponents) => shape (K, N) complex
    comp_vals = torch.exp(exponents)

    # Multiply by alpha_k: shape (K,) => broadcast with (K, N)
    # => shape (K, N)
    comp_vals = comp_vals * alpha.unsqueeze(-1)  # broadcast along N dimension

    # Sum over K => shape (N,)
    response = comp_vals.sum(dim=0)

    # If the original input was (D,) (i.e. no batch), return a scalar:
    if N == 1:
        return response[0]
    else:
        return response


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
    fig = plt.figure()
    plt.imshow(torch.sigmoid(mygrid.grid.squeeze()[:, :, grid_size // 2].detach().cpu()).numpy())
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

    bbox_length = bbox_max - bbox_min
    bbox_mean = (bbox_max + bbox_min) / 2
    bbox_min = bbox_mean - bbox_length * 2
    bbox_max = bbox_mean + bbox_length * 2

    return bbox_min, bbox_max
