# SIMSHIFT: A Benchmark for Adapting Neural Surrogates to Distribution Shifts

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2506.12007-b31b1b.svg)](https://arxiv.org/abs/2506.12007)
[![Docs](https://img.shields.io/readthedocs/simshift/latest)](https://simshift.readthedocs.io/en/latest/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.6-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/docs/2.6/)
[![hydra](https://img.shields.io/badge/Hydra_1.3-89b8cd)](https://hydra.cc/)

![Figure 1](./res/figure_1.png)

</div>

> Paper: https://arxiv.org/abs/2506.12007
>
> Authors: Paul Setinek, Gianluca Galletti, Thomas Gross, Dominik Schnürer, Johannes Brandstetter, Werner Zellinger

## Overview

This repository contains the datasets, dataloaders, baseline models, unsupervised domain adatpation algorithms and model selection strategies together with experiments and evaluation protocols for **SIMSHIFT**, a benchmark designed to evaluate Unsupervised Domain Adaptation (UDA) methods for neural surrogates of physical simulations. The benchmark's datasets target real world industrial scenarios and provides distribution shifts across parameter configurations in mesh-based PDE simulations.

**Datasets:**

- **Hot rolling**
- **Sheet metal forming**
- **Electric motor design**
- **Heatsink desing**

All the datasets are hosted on Huggingface at [https://huggingface.co/datasets/simshift/SIMSHIFT_data](https://huggingface.co/datasets/simshift/SIMSHIFT_data).

### 1. Installation

Clone the repo:

```bash
git clone https://github.com/psetinek/simshift.git
cd simshift
```

Create a new virual environment (the code was developed and tested with python 3.11):

```bash
conda create -n simshift python=3.11
conda activate simshift
```

First please install you desired torch version (the repo was tested with torch 2.6.0), as shown [here](https://pytorch.org/get-started/locally/). If you are on a linux system and have cuda 12.6, the command would be:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

Additionally, we use [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/), please install it as follows:

```bash
pip install torch_geometric
```

We also need `torch-scatter`. To install it, first check your PyTorch and CUDA version:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

Then install the packages above as described in the respective [documentations](https://pypi.org/project/torch-scatter/). For torch 2.6.0 and CUDA 12.6, it would work like:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

Then install torch-cluster with:
```bash
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

Finally, install the `simshift` package, via:

```
pip install -e .
```

#### 2. Tutorial notebooks

To get familiar with the capabilities of this repository, we provide a [`tutorial notebook`](./notebooks/tutorial.ipynb) showing easy model training and result evaluation and visualization. [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/psetinek/simshift/blob/main/notebooks/tutorial.ipynb)

Please also take a look at the [documentation](https://simshift.readthedocs.io/en/latest/) of the package.
There, we also provide instructions of how you can easily extend SIMSHIFTs current functionalities, i.e. add a new model, dataset or domain adaptation algorithm.

#### 3. Custom domain splits
In the paper, we report results on 1D/2D splits, but you can define **arbitrary n-D splits** on the respective parameters of the SIMSHIFT datasets. We provide ```domain_splitter.py```, which filters source samples by ranges on any subset of parameters, and puts everything else into target. Both domains are then split into train/(val)/test as requested.

You can use it as follows:
```bash
python domain_splitter.py \
    --data-path <path_to_dataset> \
    --config-file domain_splitting_configs/motor.json \
    --output-name splits.json
```

The arguments are:
| Argument | Required | Description |
|----------|----------|-------------|
| `--data-path` | Yes | Path to the dataset directory (must contain `metadata.csv`) |
| `--config-file` | Yes | JSON configuration file defining domain ranges and split ratios |
| `--output-name` | No | Output filename (default: `splits.json`) |

The splitting configs used for the results in the manuscript can be found [here](./domain_splitting_configs/). An example config could look like this:
```json
{
    "seed": 42,
    "source_ratios": [0.6, 0.2, 0.2],
    "target_ratios": [0.6, 0.4],
    "split_column": "Geometry.Rotor.dr3",
    "source_range": [0, 119],
    "target_ranges": {
        "easy": [119, 121],
        "medium": [121, 123],
        "hard": [123, 126]
    }
}
```

The required parameters are:
| Parameter | Description |
|-----------|-------------|
| `seed` | Random seed for reproducibility |
| `source_ratios` | `[train, val, test]` ratios for source domain (must sum to 1.0) |
| `target_ratios` | `[train, test]` ratios for target domain (must sum to 1.0) |
| `split_column` | Column name in `metadata.csv` to use for domain splitting |
| `source_range` | `[min, max)` range for source domain samples |
| `target_ranges` | Dict mapping difficulty names to `[min, max)` ranges |

Additional requirements are:
- The dataset directory must contain a `metadata.csv` file with:
  - A `sample_id` column identifying each sample
  - The conditioning column specified in `split_column`

#### 4. Paper results reproduciblility

We try to be as transparent as possible. Therefore, we provide clear instructions on how we obtained the papers results [here](./results/README.md).
