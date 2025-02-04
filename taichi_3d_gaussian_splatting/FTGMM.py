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
    f1 = transform_gmm_to_fourier1(gmm)
    f2 = transform_gmm_to_fourier2(gmm)
    f3 = transform_gmm_to_fourier3(gmm)
    alpha, complex_means, fourier_cov = transform_gmm_to_fourier_params(gmm)

    k = torch.as_tensor([[1, 2, 3]], dtype=alpha.dtype, device=alpha.device)
    v1 = f1(k)
    v2 = f2(k)
    v3 = f3(k)
    # v4 = fourier_response_at_k(k, alpha, complex_means, fourier_cov)
    return v1, v2, v3


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
    f1 = transform_gmm_to_fourier1(
        gmm,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        grid_size=grid_size,
        scale_by_voxel_volume=True  # so amplitude is in the same ballpark
    )

    # 4) Evaluate f1 at the same "shifted" frequency indices:
    #    After fftshift, the frequency index (kx,ky,kz) = 0 is in the center of volume_fft.
    #    The indices range in [-N//2, ..., N//2 - 1].
    #    We'll build a meshgrid of those indices and call f1.
    freq_indices = torch.arange(-grid_size // 2, grid_size // 2, device=device)
    kx, ky, kz = torch.meshgrid(freq_indices, freq_indices, freq_indices, indexing='ij')
    # shape: (N, N, N, 3)
    k_indices = torch.stack([kx, ky, kz], dim=-1)

    # 5) Compute the analytic GMM Fourier transform f1(k) on all these indices:
    #    We'll flatten k_indices and process in batches to avoid OOM:
    k_indices_flat = k_indices.reshape(-1, 3)  # (N^3, 3)
    batch_size = 1000  # Process 1k points at a time
    f1_values_list = []

    for i in range(0, k_indices_flat.shape[0], batch_size):
        batch = k_indices_flat[i:i + batch_size]
        f1_values_batch = f1(batch)  # shape (batch_size,) complex
        f1_values_list.append(f1_values_batch)

    # Concatenate batches
    f1_values_flat = torch.cat(f1_values_list)
    # reshape back
    f1_values = f1_values_flat.reshape(grid_size, grid_size, grid_size)

    # 6) Now compare magnitude (and/or phase) to volume_fft:
    #    For example, compute an L2 error or correlation.
    #    NOTE: volume_fft is shape (N,N,N) complex, same for f1_values.
    #    a) Magnitude difference
    diff = torch.abs(volume_fft - f1_values)
    mag_err = diff.mean().item()

    #    b) Possibly measure correlation in magnitude or real part
    corr_num = torch.sum((volume_fft.real * f1_values.real) + (volume_fft.imag * f1_values.imag))
    corr_den = torch.sqrt(
        torch.sum(volume_fft.real ** 2 + volume_fft.imag ** 2) * torch.sum(f1_values.real ** 2 + f1_values.imag ** 2))
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
    plt.imshow(torch.abs(f1_values[:, :, mid]).cpu().numpy())
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return mag_err, corr


def transform_gmm_to_fourier1(gmm, bbox_min, bbox_max, grid_size, scale_by_voxel_volume=True):
    """
    Returns a function f1(k) that computes the continuous Fourier transform
    of the GMM at the discrete frequency index k (matching torch.fft.fftn convention),
    including the global phase shift from bbox_min.

    Args:
        gmm:         A MixtureSameFamily distribution (3D).
        bbox_min:    (3,) tensor with minimum bounding box coords.
        bbox_max:    (3,) tensor with maximum bounding box coords.
        grid_size:   Integer number of voxels in each dimension (N).
        scale_by_voxel_volume:
                     If True, multiply the transform by (Δ^3) so that the amplitude
                     matches the discrete-sum-of-volume convention.

    Returns:
        f1(k): A callable that takes a tensor of shape (..., 3) for frequency indices
               in [0..N-1] or negative/positive shifts.  It returns a complex tensor
               of shape (...).
    """
    mixture_dist = gmm.mixture_distribution  # Categorical(...), shape: [num_components,]
    component_dist = gmm.component_distribution  # MultivariateNormal, shape: [num_components, ...]
    weights = mixture_dist.probs  # (num_components,)
    means = component_dist.mean  # (num_components, 3)
    covs = component_dist.covariance_matrix  # (num_components, 3, 3)

    # We define Δ per dimension (assuming isotropic grid_size for each axis).
    # shape (3,)
    delta = (bbox_max - bbox_min) / (grid_size - 1)

    def f1(k_index):
        """
        Compute the GMM's 'DFT-like' Fourier transform at the discrete frequency index k_index.

        k_index: shape [..., 3], each entry in range [-(N//2)..(N-1)//2] or [0..N-1].

        Return: shape [...], complex-valued
        """
        # Ensure k_index is float (for broadcast) and on the same device as means
        k_index = k_index.to(means.device, means.dtype)

        # Convert k_index -> physical frequency (angular freq) = omega
        #   If DFT exponent is exp[-i 2π (k·n / N)], then
        #   in continuous space with x = bbox_min + n*Δ,
        #   angular frequency is: omega = 2π * (k / N) * (1/Δ)
        # We'll do this dimension-wise.
        # shape [..., 3]
        omega = (2.0 * torch.pi) * (k_index / float(grid_size)) / delta

        # Expand shapes for broadcasting:
        #   means:         (num_components, 3)
        #   covs:          (num_components, 3, 3)
        #   weights:       (num_components,)
        #   omega:         (..., 3)
        # We want to compute for each component:
        #     exp[- i omega·(mean - bbox_min)] * exp[- 0.5 * (omega^T cov omega)]
        # then weighted sum.

        # shape [..., 1, 3]
        omega_unsq = omega.unsqueeze(-2)  # add component dimension next
        # shape (1, num_components, 3)
        means_unsq = means.unsqueeze(0)
        # Also shift the mean by bbox_min to align the "voxel index 0" with bbox_min
        means_shifted = means_unsq - bbox_min  # shape (1, num_components, 3)

        # (1, num_components, 3, 3)
        covs_unsq = covs.unsqueeze(0)
        # (1, num_components)
        weights_unsq = weights.unsqueeze(0)

        # -- Phase from the mean shift:  exp[- i (omega · (mean - bbox_min))]
        #    shape: [..., 1, num_components]
        phase_exponent = -1j * torch.sum(omega_unsq * means_shifted, dim=-1)
        phase_factor = torch.exp(phase_exponent)

        # -- Gaussian attenuation: exp[- 0.5 * omega^T cov_i omega]
        #    shape of cov_i: (1, num_components, 3, 3)
        #    shape of omega_unsq: [..., 1, 3] -> need [..., 1, 3, 1] for matmul
        omega_vec = omega_unsq.unsqueeze(-1)  # [..., 1, 3, 1]
        # shape [..., 1, num_components, 3, 1]
        cov_omega = torch.matmul(covs_unsq, omega_vec)
        # shape [..., 1, num_components, 1, 1]
        quad_form = torch.matmul(omega_vec.transpose(-2, -1), cov_omega)
        quad_form = quad_form.squeeze(-1).squeeze(-1)  # [..., 1, num_components]
        gauss_factor = torch.exp(-0.5 * quad_form)

        # Weighted sum over components
        # shape [..., 1, num_components]
        w_factor = weights_unsq * phase_factor * gauss_factor
        # shape [...]
        ft_value = w_factor.sum(dim=-1)

        # Optionally scale by voxel volume so the amplitude matches the discrete-sum
        if scale_by_voxel_volume:
            vol = torch.prod(delta)  # Δ^3
            ft_value = ft_value * vol

        return ft_value

    return f1


def transform_gmm_to_fourier2(gmm: MixtureSameFamily):
    """
    Transform GMM (MixtureSameFamily of MultivariateNormal) into its
    complex-valued Fourier transform, using the closed-form expression.

    Args:
        gmm (MixtureSameFamily):
            A mixture of MultivariateNormal distributions.

    Returns:
        fourier_gmm: A callable that takes a frequency vector k (size D or [...,D])
                     and returns the complex value of the Fourier transform.
        weights: A tensor of shape (K,) with mixture weights.
        locs: A tensor of shape (K, D) with means of components.
        covs: A tensor of shape (K, D, D) with covariance matrices of components.

        (You can think of fourier_gmm(k) as sum_k w_k * exp(- i mu_k^T k - 1/2 k^T Sig_k k).)
    """
    # -- Extract GMM parameters --
    # Mixture weights (K,)
    weights = gmm.mixture_distribution.probs  # shape: (K,)

    # Means (K, D)
    locs = gmm.component_distribution.loc  # shape: (K, D)

    # Covariance matrices (K, D, D)
    covs = gmm.component_distribution.covariance_matrix  # shape: (K, D, D)

    # -- Build the callable for the Fourier transform --
    def fourier_gmm(k: torch.Tensor) -> torch.Tensor:
        """
        Compute the complex Fourier transform of the GMM at frequency k.
        
        Args:
            k: A 1D tensor of shape (D,) or a batch of shape (..., D).
        
        Returns:
            A complex tensor of shape (...) giving the Fourier transform at k.
        """
        # Convert k to tensor if not already
        k = torch.as_tensor(k, dtype=locs.dtype, device=locs.device)

        # Make sure k has a final dimension D, but allow leading batch dimensions.
        # E.g. k could be shape (D,) or (B, D).
        # We'll broadcast accordingly.
        # We'll accumulate the sum across mixture components in a vectorized manner if possible.

        # k shape: (..., D)
        # locs shape: (K, D)
        # covs shape: (K, D, D)
        # We want to compute for each mixture component:
        #   w_k * exp( - i * loc_k^T * k - 1/2 * k^T * Sigma_k * k )
        # in a broadcast-friendly way.

        # Expand k so that we can batch-multiply by covs. We want something like
        #    (K, ..., D) x (K, D, D) x (K, ..., D)
        # We'll do it carefully below.

        # First, check if k has batch dims
        # We'll reshape so we broadcast over mixture components in a new dimension.
        #   => k shape becomes (1, ..., D)
        # This means the mixture dimension = K can broadcast over that "1".
        k_batched = k.unsqueeze(dim=0)  # shape: (1, ..., D)

        # Step 1: Quadratic form  (k^T Sigma_k k) for each mixture component
        # We can do:
        #   diff = k_batched  # shape: (1, ..., D)
        #   covs[i] shape: (D, D)
        # but we have K of those, so covs shape is (K, D, D).
        # We'll use a batch matmul approach.

        # We want something like: (K, ..., D) * (K, D, D) * (K, ..., D).
        # We'll expand k_batched to (K, ..., D) by repeating along the first dimension K times:
        k_expanded = k_batched.expand(weights.shape[0], *k_batched.shape[1:])  # (K, ..., D)

        # Now do:  (K, ..., 1, D) * (K, ..., D, D) => (K, ..., 1, D)
        # to get the product k^T Sigma_k
        # We'll need an extra unsqueeze for proper matmul dimensions:
        k_expanded_2 = k_expanded.unsqueeze(-2)  # (K, ..., 1, D)
        covs_expanded = covs.unsqueeze(1).expand(-1, k_expanded.shape[1], -1, -1)
        # covs_expanded now shape: (K, ..., D, D) (broadcasting in the same batch dims as k)

        # Multiply: (K, ..., 1, D) x (K, ..., D, D) => (K, ..., 1, D)
        kT_Sigma = torch.matmul(k_expanded_2, covs_expanded)  # shape: (K, ..., 1, D)

        # Finally multiply kT_Sigma by k again: (K, ..., 1, D) x (K, ..., D) => (K, ..., 1)
        # We'll need another unsqueeze for the second factor:
        k_expanded_3 = k_expanded.unsqueeze(-1)  # shape: (K, ..., D, 1)
        quad_form = torch.matmul(kT_Sigma, k_expanded_3)  # shape: (K, ..., 1, 1)

        # quad_form is (K, ..., 1, 1). Squeeze out the last two dims => (K, ..., )
        quad_form = quad_form.squeeze(-1).squeeze(-1)  # shape: (K, ...)

        # Step 2: Linear term  ( - i loc_k^T k )
        # We'll do loc_k^T k similarly:
        # locs shape: (K, D)
        # k_expanded shape: (K, ..., D)
        # We'll just do a dot product across D:
        # => (K, ..., )
        linear_term = (locs.unsqueeze(1) * k_expanded).sum(dim=-1)  # shape: (K, ...,)

        # Step 3: Combine exponent
        # exponent = -1/2 * quad_form + (-i) * linear_term
        # We'll form a complex tensor in PyTorch by using `1j` for the imaginary part:
        exponent = -0.5 * quad_form + (-1j) * linear_term  # shape: (K, ...)

        # Step 4: Multiply by mixture weights and sum
        # weights shape: (K,)
        # We want to broadcast that across all the leading dimensions of exponent.
        w = weights.view(-1, *([1] * exponent.ndim)[1:])  # reshape (K, 1, 1, ...) if needed
        # but more simply we can do a direct broadcast:
        w = weights[:, None] if exponent.ndim > 1 else weights  # shape (K, ...) 
        # The exponent is (K, ...). So we can do exp(exponent) => (K, ...)
        # Then multiply by w => (K, ...)
        # Then sum over K => (...)

        out = w * torch.exp(exponent)
        fourier_vals = out.sum(dim=0)  # sum across the mixture dimension

        return fourier_vals  # shape: (...), complex

    return fourier_gmm


def transform_gmm_to_fourier3(gmm: MixtureSameFamily):
    """
    Transform a multivariate Gaussian mixture model (GMM) into Fourier space
    using the closed-form equation of the Fourier transform of Gaussians.

    The returned object is a callable that, given wavevector(s) k, returns
    the (complex) values of the GMM's Fourier transform at k.

    Args:
        gmm (MixtureSameFamily): A GMM where the component_distribution
            is a MultivariateNormal, and mixture_distribution is a Categorical.

    Returns:
        A callable fourier_transform(k: Tensor) -> Tensor, where the shape of
        k can be (..., d), and the result will be (...,) (a complex tensor).
    """
    # Extract mixture weights of shape (num_components,)
    weights = gmm.mixture_distribution.probs

    # Extract means of shape (num_components, d)
    # or possibly (..., num_components, d) if batchified.
    # For simplicity, we assume (num_components, d).
    means = gmm.component_distribution.loc

    # We will extract either the covariance_matrix or the scale_tril.
    # If you have 'covariance_matrix' for each component,
    # it will be of shape (num_components, d, d).
    # If the distribution is using 'scale_tril', you can convert it
    # to covariance via: Sigma = scale_tril @ scale_tril.T
    covs = gmm.component_distribution.covariance_matrix

    def fourier_transform(k: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the (complex) Fourier transform of the GMM at wavevector(s) k.

        Args:
            k (Tensor): Wavevector of shape (..., d), where d is the dimensionality
                        of the GMM. The "..." can be any shape, including none
                        for a single wavevector of shape (d,).

        Returns:
            Tensor: Complex tensor of shape (...,) giving the GMM's FT at each k.
        """
        # Ensure k has shape (..., d). If k is (d,), unsqueeze to (1, d).
        # This way we handle a batch of k or a single k uniformly.
        if k.dim() == 1:
            k = k.unsqueeze(0)  # becomes (1, d)

        # We will accumulate the transform over all mixture components.
        # Output shape: (...,) => same leading shape as k, after summation over components.
        # We'll store partial sums in complex form:
        ft_sum = torch.zeros(k.shape[:-1], dtype=torch.complex64, device=k.device)

        # Loop over the components
        for weight_i, mean_i, cov_i in zip(weights, means, covs):
            # 1) Compute the exponent for the Gaussian part:
            #    -1/2 * (k^T * Σ * k), for each wavevector in batch
            # shape of mean_i: (d,)
            # shape of cov_i: (d, d)
            # shape of k: (..., d)

            # Expand mean_i to broadcast with k:
            # We'll do a dot product k * mean_i along the last dim (d).
            # For the quadratic form k^T Σ k:
            #   quadratic = sum over d of (k^T Σ k).
            # An efficient way is `torch.einsum('...i,ij,...j->...', k, cov_i, k)`.

            quad_form = 0.5 * torch.einsum('...i,ij,...j->...', k, cov_i, k)  # shape (...,)

            # 2) Real part exponent: exp(-quad_form)
            real_factor = torch.exp(-quad_form)

            # 3) Complex phase: exp(- i * k^T mean_i)
            #    The dot k^T mean_i is sum over d of (k[..., d] * mean_i[d]).
            #    We'll do `torch.einsum('...i,i->...', k, mean_i)`.

            phase = -1.0j * torch.einsum('...i,i->...', k, mean_i)  # shape (...,)
            complex_phase = torch.exp(phase)  # shape (...,) (complex)

            # 4) Full component FT = exp(-1/2 k^T Σ k) * exp(-i k^T mu)
            component_ft = real_factor * complex_phase

            # 5) Weight by the mixture probability
            ft_sum = ft_sum + weight_i * component_ft

        return ft_sum

    return fourier_transform


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
