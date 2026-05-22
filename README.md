# CNN+FoF: application of deep learning to the identification of dark matter haloes

[![Paper](https://img.shields.io/badge/arXiv-2602.21246-b31b1b.svg)](https://arxiv.org/abs/2602.21246)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A hybrid deep learning pipeline for identifying dark matter haloes in cosmological N-body simulations. A volumetric 3D Convolutional Neural Network classifies individual simulation particles as halo or non-halo members, followed by a highly optimised and parallelised Friends-of-Friends (FoF) clustering algorithm that groups the classified particles into distinct halo objects.

> **Maiti S., Correa C. M., Fiorilli A., Ruiz A. N., Paz D. J., Pérez Fernández A., Sánchez A. G.**  
> *CNN+FoF: application of deep learning to the identification of dark matter haloes*, MNRAS (2026)

---

## Overview

Traditional halo finders are CPU-based and iterative, becoming a severe bottleneck in modern GPU-accelerated cosmological pipelines. CNN+FoF addresses this by:

- Performing binary particle classification in a **single GPU-native forward pass**
- Reducing the particle search space by ~63% before the FoF stage
- Achieving a **~10× speed-up** over ROCKSTAR across all tested resolutions
- Recovering halo catalogues with **>98% classification accuracy** and **>95% purity**
- Reproducing the halo mass function to within **5%** of the ROCKSTAR reference


---

## Requirements

```
Python      >= 3.10
PyTorch     >= 2.0
NumPy
SciPy
pandas
Matplotlib
```

Install dependencies:

```bash
pip install torch numpy scipy pandas matplotlib
```

Build the FoF executable:

```bash
cd voxcel_fof/
make
```

---

## Data

Simulations were generated with [GADGET-4](https://gitlab.mpcdf.mpg.de/vspringe/gadget4) assuming a flat ΛCDM cosmology at $z=0$. Ground-truth halo labels were obtained from [ROCKSTAR](https://github.com/yt-project/rockstar-galaxies).

Four resolution configurations were used:

| Configuration  | $m_p \ [M_\odot]$      | $N_\mathrm{particles}$ |
|----------------|------------------------|------------------------|
| L200-N32³      | $4.35 \times 10^{12}$  | $32{,}768$             |
| L200-N64³      | $5.44 \times 10^{11}$  | $262{,}144$            |
| L200-N128³     | $6.80 \times 10^{10}$  | $2{,}097{,}152$        |
| L100-N128³     | $8.50 \times 10^{9}$   | $2{,}097{,}152$        |

The input to the network is a 6-channel voxelised tensor per particle:

$$X \in \mathbb{R}^{N_\mathrm{res} \times N_\mathrm{res} \times N_\mathrm{res} \times 6}$$

comprising the three displacement field components $(\Psi_x, \Psi_y, \Psi_z)$ and three velocity components $(v_x, v_y, v_z)$.

---

## Model Architecture

The classification network is a 3D VNet (encoder–decoder with skip connections), adapted for binary particle classification:

```
Input (6 channels)
    │
    ├─ Conv3D 6→64  ×2          [stride 1, periodic padding]
    │       │
    ├─ Conv3D 64→128            [stride 2, downsample]
    │       │
    ├─ Conv3D 128→128  ×2
    │       │
    ├─ Conv3D 128→256           [stride 2, downsample]
    │       │
    ├─ Conv3D 256→256  ×2       [bottleneck]
    │       │
    ├─ ConvTranspose3D 256→128  [upsample + skip connection]
    │       │
    ├─ Conv3D 256→128  ×2       [after concat]
    │       │
    ├─ ConvTranspose3D 128→64   [upsample + skip connection]
    │       │
    ├─ Conv3D 128→64  ×2        [after concat]
    │       │
    └─ Conv3D 64→1 + Sigmoid
            │
        Output: halo probability ∈ [0, 1] per particle
```

| Parameter           | Value                             |
|---------------------|-----------------------------------|
| Input channels      | 6 (displacement + velocity)       |
| Output channels     | 1 (halo probability)              |
| Trainable params    | ~8.4 × 10⁶                        |
| Loss function       | Binary cross-entropy (BCE)        |
| Optimiser           | Adam, lr = 10⁻³                   |
| Max epochs          | 120 (early stopping on val. loss) |
| Classification threshold | 0.498                        |


## Results

### Particle classification (L100-N128³)

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 0.99   |
| Precision   | 0.98   |
| Recall      | 0.98   |
| F1 Score    | 0.98   |
| AUC         | 0.99   |

### Halo catalogue (L100-N128³, $M_{200b}$)

| Metric       | Value   |
|--------------|---------|
| Purity       | 99.38%  |
| Completeness | 89.34%  |
| HMF agreement| < 5%    |

### Computational speed-up

The CNN+FoF pipeline achieves a consistent speed-up of approximately one order of magnitude relative to ROCKSTAR across all tested resolutions, with runtime ratios ranging from 8× to 12×.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{maiti2026cnnfof,
  title   = {{CNN+FoF}: application of deep learning to the identification of dark matter haloes},
  author  = {Maiti, Soumadeep and Correa, Carlos M. and Fiorilli, Andrea and
             Ruiz, Andr\'es N. and Paz, Dante J. and
             P\'erez Fern\'andez, Alejandro and S\'anchez, Ariel G.},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2026},
  eprint  = {2602.21246},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO}
}
```

---

## Acknowledgements

This work was carried out on the HPC system Raven at the [Max Planck Computing and Data Facility (MPCDF)](https://www.mpcdf.mpg.de) in Garching, Germany. Supported by the Excellence Cluster ORIGINS, funded by the Deutsche Forschungsgemeinschaft (DFG) under Germany's Excellence Strategy — EXC-2094 — 390783311.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.