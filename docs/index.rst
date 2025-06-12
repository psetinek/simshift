SIMSHIFT
========

.. image:: https://github.com/psetinek/simshift/blob/main/res/figure_1.png?raw=true
   :alt: simshift_figure1.png


What is ``SIMSHIFT``?
---------------------

SIMSHIFT is a **benchmark designed to evaluate Unsupervised Domain Adaptation (UDA)** methods for neural surrogates of physical simulations.
In particular, it targets real world industrial scenarios and provides pre-defined distribution shifts across parameter configurations in mesh-based PDE simulations.

The library contains dataloaders, baseline models, unsupervised domain adatpation algorithms and model selection strategies. 


.. note::

   For more details on SIMSHIFT, check out our `preprint <https://arxiv.org/pdf/TODO>`_.



Datasets
--------
SIMSHIFT includes four practical datasets, with predefined distribution shifts. All datasets are publicly hosted on `huggingface <https://huggingface.co/datasets/simshift/SIMSHIFT_data>`_.


1. **Hot Rolling:** a metal slab plastically deformed into a sheet metal product. 

2. **Sheet Metal Forming:** a sheet metal supported at the ends and center, a holder and a punch deforms.

3. **Electric Motor:** a structural FEM simulation of a rotor in electric machinery, subjected to mechanical loading at burst speed.

4. **Heatsink:** CFD simulation focused on the thermal performance of heat sinks, commonly used in electronic cooling applications.

+-----------+--------------+---------+------------------+-----------------+--------------------------+------+-------+
| Dataset   | Origin       | Samples | Output channels  | Avg. # nodes    | Varied simulation params | Dim  | (GB)  |
+===========+==============+=========+==================+=================+==========================+======+=======+
| Rolling   | Metallurgy   | 4,750   | 10               | 576             | 4                        | 2D   | 0.5   |
+-----------+--------------+---------+------------------+-----------------+--------------------------+------+-------+
| Forming   | Manufacturing| 3,315   | 10               | 6,417           | 4                        | 2D   | 4.1   |
+-----------+--------------+---------+------------------+-----------------+--------------------------+------+-------+
| Motor     | Machinery    | 3,196   | 26               | 9,052           | 15                       | 2D   | 13.4  |
+-----------+--------------+---------+------------------+-----------------+--------------------------+------+-------+
| Heatsink  | Electronics  | 460     | 5                | 1,385,594       | 4                        | 3D   | 40.8  |
+-----------+--------------+---------+------------------+-----------------+--------------------------+------+-------+


Models
------
SIMSHIFT includes four machine learning models, commonly used in the field of AI for simulation.

1. **PointNet** [`Qi et al., 2017. <https://arxiv.org/abs/1612.00593>`_] integrates global context by aggregating local features from all input points into a shared global representation.

2. **GraphSAGE** [`Hamilton et al., 2017. <https://arxiv.org/abs/1706.02216>`_] is a graph neural network that captures local information via message passing, suited for complex meshes but can be computationally expensive.

3. **Transolver** [`Wu et al., 2024. <https://arxiv.org/abs/2402.02366>`_] is a state of the art Transformer-based model with Physics-Attention, to capture complex geometries and long-range interactions.

4. **UPT** [`Alkin et al., 2024. <https://arxiv.org/abs/2402.12365>`_] is a state of the art neural operator with a focus on scalability, that represents fields in a latent space and directly learns latent dynamics.


UDA Methods
-----------
SIMSHIFT implements several unsupervised domain adaptation (UDA) methods to address distribution shifts between source and target simulation domains.

1. **Correlation Alignment (DeepCORAL)** [`Sun and Saenko, 2016. <https://arxiv.org/abs/1607.01719>`_]
   aligns the covariance of the source and target feature representations.

2. **Central Moment Discrepancy (CMD)** [`Zellinger et al., 2017. <https://arxiv.org/abs/1702.08811>`_]
   aligns central moments (mean, variance, skewness, etc.) of the source and target feature representations.

3. **Domain-Adversarial Neural Network (DANN)** [`Ganin et al., 2016. <https://arxiv.org/abs/1505.07818>`_]
   introduces a domain classifier and adversarial loss that encourage the feature encoder to learn domain-invariant features.


Model Selection
---------------
SIMSHIFT supports several model selection strategies for unsupervised the unsupervised setting (no labels in the target domain).

1. **Deep Embedded Validation (DEV)** [`You et al., 2019. <https://proceedings.mlr.press/v97/you19a.html>`_]
   selects the model with the lowest variance in prediction consistency across nearby samples in the target domain, using the idea that robust models produce smooth outputs.

2. **Importance Weighted Validation (IWV)** [`Sugiyama et al., 2007. <https://jmlr.org/papers/v8/sugiyama07a.html>`_]
   estimates model performance on the target domain by reweighting source validation loss using learned importance weights between source and target feature distributions.

3. **Source Best (SB)**
   chooses the model with the lowest average validation loss on the source domain. This is a naive baseline that assumes source domain performance correlates with target performance.

4. **Target Best (TB)**
   selects the best model per sample using ground truth target losses (oracle). This method is not available during real world deployment but serves as an upper bound for model selection performance.

Each model selection algorithm in SIMSHIFT returns a weight vector over candidate models and can be plugged into ensemble evaluation or winner-takes-all prediction modes.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   pages/installation
   pages/tutorials
   pages/extending_simshift

.. toctree::
   :maxdepth: 2
   :caption: API

   pages/data
   pages/da_algorithms
   pages/models
   pages/model_selection