Installation
============

To install the SIMSHIFT package, please follow the step-by-step instructions below.

**1. Clone the repository**

.. code-block:: bash

    git clone <path_to_repo_upon_publication>
    cd simshift

**2. Create a virtual environment**

We recommend using Python ``3.11``, which was used during development and testing.

.. code-block:: bash

    conda create -n simshift python=3.11
    conda activate simshift

**3. Install PyTorch**

First, install your desired version of PyTorch. SIMSHIFT was tested with PyTorch ``2.6.0``.

Follow the official [`PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_].

For example, on Linux with CUDA ``12.6``:

.. code-block:: bash

    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

**4. Install PyTorch Geometric (PyG)**

SIMSHIFT depends on [`PyTorch Geometric <https://pytorch-geometric.readthedocs.io/>`_]. You can install the core package with:

.. code-block:: bash

    pip install torch_geometric

**5. Install torch-scatter**

The package torch-scatter needs to match your installed PyTorch and CUDA versions.

To check your current setup:

.. code-block:: bash

    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

Then install the appropriate version following the instructions [`here <https://pypi.org/project/torch-scatter/>`_].

For PyTorch ``2.6.0`` and CUDA ``12.6``, use:

.. code-block:: bash

    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

**6. Install SIMSHIFT**

Finally, install the simshift package in editable mode:

.. code-block:: bash

    pip install -e .
