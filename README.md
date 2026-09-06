# Geometry-Preserving in 3D Gaussian Splatting for LiDAR-Camera Extrinsic Calibration

Official implementation of **GeoP-Calib**: *Geometry-Preserving in 3D Gaussian Splatting for LiDAR-Camera Extrinsic Calibration*.

[![Paper](https://img.shields.io/badge/arXiv-2606.20103-b31b1b.svg)](https://arxiv.org/abs/2606.20103)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://airlabkhu.github.io/GeoP-Calib/)

**[Paper](https://arxiv.org/abs/2606.20103)** | **[Project Page](https://airlabkhu.github.io/GeoP-Calib/)**

Kyoleen Kwak, Daeho Kim, Jeong Woon Lee, Hyoseok Hwang

---

## Setup

Tested on Ubuntu 20.04 with CUDA 12.6, PyTorch 2.9.0 and an NVIDIA RTX 4070 Ti.

Clone the repository together with its submodules:

```shell
git clone --recursive https://github.com/AIRLABkhu/GeoP-Calib.git
cd GeoP-Calib
```

Create the conda environment:

```shell
conda env create -f environment.yml
conda activate GeoP-Calib
```

Build the CUDA extensions. `--no-build-isolation` is required so that the
extensions compile against the PyTorch installed in the environment:

```shell
pip install --no-build-isolation \
    submodules/diff-gaussian-rasterization \
    submodules/simple-knn \
    submodules/fused-ssim
```

Install PyTorch3D from source:

```shell
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

### RoMa weights

GeoP-Calib uses [RoMa](https://github.com/Parskatt/RoMa) for inter-frame flow extraction.
Download `roma_outdoor.pth` and `dinov2_vitl14_pretrain.pth` following the official RoMa
implementation and place both files in a single directory, which is then passed as
`--flow_path`.

## Dataset

> **Coming soon.** Dataset preparation instructions and preprocessed sequences will be released shortly.

## Running

To run the calibrator:

```shell
python calibrate_GeoP.py -s <path_of_the_dataset> --flow_path <path_of_the_flow_model_weights> --optimizer_type sparse_adam --cam_id X --data_seq X
```

### KITTI-360

```shell
python calibrate_GeoP.py -s dataset/KITTI360 --flow_path weights/ --optimizer_type sparse_adam --cam_id 00 --data_seq 2
```

Here, `--cam_id` should be either `00` or `01`, and `--data_seq` should be one of `0` to `4`.

### KITTI

For the KITTI dataset, change the dataset path accordingly and use the fixed setting `--cam_id 01 --data_seq 0 --dataset_type KITTI`:

```shell
python calibrate_GeoP.py -s dataset/KITTI/5-300-t --flow_path weights/ --optimizer_type sparse_adam --cam_id 01 --data_seq 0 --dataset_type KITTI
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{kwak2026geometry,
  title={Geometry-Preserving in 3D Gaussian Splatting for LiDAR-Camera Extrinsic Calibration},
  author={Kwak, Kyoleen and Kim, Daeho and Lee, Jeong Woon and Hwang, Hyoseok},
  journal={arXiv preprint arXiv:2606.20103},
  year={2026}
}
```

## Acknowledgements

This work is built upon the [HiGS-Calib](https://github.com/IRMVLab/HiGS-Calib) codebase. We thank the authors for releasing their code.

If you use this repository, please also consider citing HiGS-Calib:

```bibtex
@article{zhang2025higs,
  title={Higs-calib: A hierarchical 3d gaussian splatting based targetless local-consistent lidar-camera calibration method},
  author={Zhang, Tianjun and Zhang, Lin and Wang, Hesheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

We also thank the authors of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [RoMa](https://github.com/Parskatt/RoMa), and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for their open-source contributions.
