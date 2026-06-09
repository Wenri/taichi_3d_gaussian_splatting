# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An unofficial, **pure-Python implementation of 3D Gaussian Splatting** (the INRIA radiance-field paper) built on the [Taichi](https://www.taichi-lang.org/) language instead of hand-written CUDA. This checkout is a fork (`Wenri/taichi_3d_gaussian_splatting`) of `wanmeihuali/taichi_3d_gaussian_splatting` with an added experimental Fourier/GMM analysis module (`FTGMM.py`).

Taichi runs **CUDA backend only** here. Developed against Python 3.10, RTX 3090, CUDA 12.1.

## Commands

```bash
# Install (torch/torchvision are expected to be installed already, usually via conda)
pip install -r requirements.txt
pip install -e .
pip install pytorch3d          # REQUIRED but missing from requirements.txt — see Gotchas

# Train (everything is driven by a single YAML config)
python gaussian_point_train.py --train_config config/tat_truck_every_8_test.yaml

# Emit a fully-populated config template with defaults
python gaussian_point_train.py --train_config out.yaml --gen_template_only

# Render a trained scene (.parquet) along a camera path (.pt of 4x4 SE3 matrices, or a dataset .json)
python gaussian_point_render.py --parquet_path scene.parquet --poses poses.json --output_prefix out/

# Interactive Taichi-GUI viewer (merges multiple point clouds into one scene)
python visualizer.py --parquet_path_list a.parquet b.parquet

# Tests (unittest; pattern is *_test.py under tests/). Require a CUDA GPU — they launch Taichi kernels.
python -m unittest discover -s tests -p '*test.py'
python -m unittest tests.utils_test          # single test module
```

Data prep scripts live in `tools/`: `prepare_colmap.py`, `prepare_InstantNGP_with_mesh.py` (samples a point cloud from a mesh for Instant-NGP/BlenderNeRF datasets), `prepare_kitti.py`. The README has detailed dataset-by-dataset instructions.

## Architecture

The training loop in `GaussianPointTrainer.py` wires four cooperating modules around a Taichi rasterizer:

1. **`GaussianPointCloudScene`** — an `nn.Module` holding the optimizable state as two `nn.Parameter`s: `point_cloud` (Nx3 xyz positions) and `point_cloud_features` (Nx56). Plus non-trainable buffers `point_invalid_mask` (N, int8) and `point_object_id` (N, int32). Loads/saves `.parquet` (and `.ply`). On load it pre-allocates extra rows (`max_num_points_ratio`) and marks them invalid so densification can activate points **without reallocating**.

2. **`GaussianPointCloudRasterisation`** — the heart of the repo, and the file you'll spend the most time in. It is a `torch.nn.Module` wrapping a custom `torch.autograd.Function`; that Function's `forward`/`backward` call hand-written Taichi `@ti.kernel`s. **Gradients are computed manually** in `gaussian_point_rasterisation_backward` — this is NOT Taichi autodiff. Forward pipeline: `filter_point_in_camera` → `generate_point_attributes_in_camera_plane` (project 3D Gaussians to 2D conics + evaluate SH color) → sort points by (tile, depth) key → `gaussian_point_rasterisation` (per-16x16-tile front-to-back alpha blending). Outputs `(image HxWx3, depth, per-pixel valid-point count)`.

3. **`GaussianPointAdaptiveController`** — adaptive density control (clone/split/prune "floaters" and transparent points). It is **coupled to the rasterizer through `backward_valid_point_hook`**: the rasterizer's backward kernel calls `adaptive_controller.update(...)` with per-point gradient stats, and the trainer then calls `adaptive_controller.refinement()` to mutate the scene's parameters/masks in place. This callback wiring is set up in `GaussianPointTrainer.__init__`.

4. **`LossFunction`** — `(1-λ)·L1 + λ·(1-SSIM)`, with optional scale regularization on `exp(s)`.

`ImagePoseDataset.py` reads a dataset JSON (`image_path`, `T_pointcloud_camera`, `camera_intrinsics`, `camera_height/width`, `camera_id`) and yields `(image, q_pointcloud_camera, t_pointcloud_camera, CameraInfo)`. **One image per training step** (`batch_size=None`); there is no batching. Images larger than 1600px are auto-downscaled, and all images are cropped to multiples of 16.

### The 56-dim feature vector

`point_cloud_features[i]` is laid out as:

| slice | meaning |
|---|---|
| `[0:4]` | covariance rotation quaternion, **xyzw** order |
| `[4:7]` | covariance scale, stored as **log**; actual axis lengths are `exp(s)` |
| `[7]` | opacity α **before sigmoid** |
| `[8:24]` | red spherical-harmonic coefficients (16 = up to band 3) |
| `[24:40]` | green SH coefficients (16) |
| `[40:56]` | blue SH coefficients (16) |

### Conventions & mechanics worth knowing

- **Camera frame:** x-right, y-down, z-forward. Poses are passed as `q_pointcloud_camera`/`t_pointcloud_camera` (SE3 split into quaternion + translation; conversions in `utils.py`).
- **`point_object_id`** lets points belong to different objects with different camera poses in one scene — this is what enables scene merging and (future) rigid-object motion. Single-scene training leaves it all 0.
- **Taichi is initialized with `device_memory_GB=0.1`** because the code uses **no Taichi fields** — all data crosses the torch↔Taichi boundary as `ti.types.ndarray` views over torch tensors (see `utils.py` `torch2ti`/`ti2torch`).
- **Progressive training:** image downsample factor is halved over time (`initial_downsample_factor`, `half_downsample_factor_interval`) and the SH band ceiling rises over time (`color_max_sh_band = iteration // increase_color_max_sh_band_interval`).
- Per-attribute gradient rescaling factors (`grad_color_factor`, `grad_s_factor`, `grad_alpha_factor`, …) live on `GaussianPointCloudRasterisationConfig` and are applied inside the backward kernel.
- **Tiles are 16x16** (`TILE_WIDTH`/`TILE_HEIGHT` in `GaussianPointCloudRasterisation.py`).

### Config system

All configuration is a single YAML deserialized via `dataclass-wizard`'s `YAMLWizard` into nested dataclasses. `GaussianPointCloudTrainer.TrainConfig` embeds `rasterisation_config`, `adaptive_controller_config`, `gaussian_point_cloud_scene_config`, and `loss_function_config`, each defined as a nested dataclass on its owning module. YAML keys accept both kebab-case and snake_case. When adding a tunable, add it to the relevant nested `@dataclass ...Config` (not to a free-floating dict).

Outputs per run: TensorBoard logs plus `scene_<iteration>.parquet` and `best_scene.parquet`, all under `output_model_dir` (defaults to `summary_writer_log_dir`).

### FTGMM.py (fork-specific, experimental)

Added in this fork (author: Bingchen Gong). Fits the live scene as a torch Gaussian Mixture Model, samples it onto a voxel grid, and runs Fourier-domain analysis, writing diagnostic plots into `vis/`. It is **wired directly into the training loop**: `ft_grab_scene(self.scene)` runs every 1234 iterations (`GaussianPointTrainer.train`). It depends on `pytorch3d`.

## Gotchas

- **`pytorch3d` is required to train but is not in `requirements.txt`.** Because `FTGMM.ft_grab_scene` is called from the main training loop, training crashes without it. Install separately, or comment out the `ft_grab_scene` call / `FTGMM` import in `GaussianPointTrainer.py` if you don't need the analysis.
- **If you change the forward rasterization math, you must update the backward kernel by hand** (`gaussian_point_rasterisation_backward`) — there is no autodiff to fall back on. Mismatched forward/backward is the classic source of silent training divergence here.
- CUDA only. There is no CPU/Metal/OpenGL path despite Taichi supporting them.

## Cloud / CI training

`.github/workflows/run_experiment.yml` triggers when a PR gets a label prefixed `need_experiment` (e.g. `need_experiment_tat_truck`). It builds `Dockerfile.aws`, pushes to AWS ECR, and launches a SageMaker training job via `ci/run_experiment.py` (g4dn.xlarge spot, NVIDIA T4). `ci/entrypoint.sh` is the container entrypoint and expects `$TRAIN_CONFIG` to point at a config under `config/`.
