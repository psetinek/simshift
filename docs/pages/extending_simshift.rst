Extending SIMSHIFT
==================

We tried to keep SHIMSHIFT as modular and extensible as possible. It is fairly easy to extend SIMSHIFT by adding custom datasets, models, domain adaptation algorithms or
unsupervised model strategies. We will walk you through some quick examples in this section.


Adding Datasets
---------------

SIMSHIFT provides a modular interface for datasets via a ``BaseDataset`` class, which all built-in datasets inherit from. This base class offers standard utilities for downloading data, loading it into memory, computing normalization statistics, and applying normalization transforms.

To add a custom dataset, you have two options:

1. **Inherit from** ``BaseDataset`` and override any relevant methods.
2. **Implement from scratch** by inheriting from ``torch.utils.data.Dataset`` if your use case deviates significantly.

We recommend reviewing the implementation in ``simshift/data/__init__.py`` (ADD LINK) for a concrete example that inherits from the ``BaseDataset`` (ADD LINK) class.

In addition to your dataset class, you must define a ``get_dataset`` function that loads and returns the dataset, along with its normalization stats. Below is a minimal working example:

.. code-block:: python

    def get_my_new_dataset(
        split: str,
        normalization_method: Literal["zscore", "minmax"] = "zscore",
        normalization_stats: Optional[Dict] = None,
        **kwargs,
    ):
        """Return a configured dataset by loading it from disk."""
        # Source domain
        dataset_source = MyNewDataset(split=split, domain="source", **kwargs)

        if split == "train":
            normalization_stats = dataset_source.get_normalization_stats(
                method=normalization_method
            )
        assert normalization_stats is not None
        dataset_source.normalization_stats = normalization_stats
        dataset_source.normalize(method=normalization_method)

        # Target domain
        dataset_target = MyNewDataset(split=split, domain="target", **kwargs)
        dataset_target.normalization_stats = normalization_stats
        dataset_target.normalize(method=normalization_method)

        return (dataset_source, dataset_target), normalization_stats

This function must be registered in the dataset builder dictionary located in ``simshift/data/__init__.py`` (ADD LINK) as follows:

.. code-block:: python

    ...
    from .my_new_dataset import get_my_new_dataset

    DATASET_BUILDERS = {
        ...,
        "my_new_dataset": get_my_new_dataset,
    }

Finally, create a configuration file named, e.g. ``my_new_dataset.yaml`` inside the dataset config directory (ADD LINK). Once this is done, your new dataset can be used in any training or evaluation pipeline by referencing it in your config.


Adding Models
-------------

To add a new model, simply create a class inheriting from ``torch.nn.Module`` in the models folder (ADD LINK) and decorate it with the `@register_model` decorator. An example template would look something like:

.. code-block:: python

   from simshift.models import register_model

   @register_model("MyNewModel")
   class MyNewModel(nn.Module):
       def __init__(self, ...):
           ...

       def forward(self, x):
           ...

Once registered, create a model config (see e.g. the example config for PointNet (ADD LINK)). It could look something like this:

.. code-block:: yaml

    name: MyModel
    hparams:
        ...
        ...

To then use your new model, simply link your model config in the main.yaml (ADD LINK) config file.


Adding Unsupervised Domain Adaptation Algorithms
------------------------------------------------

New UDA algorithms should inherit from the provided ``ccc`` (ADD LINK) class. You simply have to add any configuration hyperparameters as needed, and implement an
``_update`` method. This ``_update`` method should compute all needed losses for optimization and store them in ``self.loss`` and ``self.loss_dict``. Just as when adding new models, new UDA algorithms should registered via the ``@register_da_method`` decorator. For an example, see the ``DeepCORAL``
(ADD LINK) class. A minimal example template could look like this:

.. code-block:: python

    from simshift.da_algorithms import DAAlgorithm, register_da_algorithm


    @register_da_algorithm("deep_coral")
    class MyUDAAlgorithm(DAAlgorithm):
        def __init__(self, some_hyperparam, **base_class_kwargs):
            self.some_hyperparam = some_hyperparam
            super().__init__(**base_class_kwargs)

        def _update(self, src_sample, trgt_sample, **kwargs):
            _ = kwargs
            # predictions
            src_pred, src_latents = self.model(**src_sample.as_dict())
            src_pred, pred_coords = src_pred
            _, trgt_latents = self.model(**trgt_sample.as_dict())

            # positions loss
            pos_loss = self.mse_loss(pred_coords, src_sample.y_mesh_coords)

            # prediction loss
            mse_loss = self.mse_loss(src_pred, src_sample.y)

            # new da loss
            da_loss = self._my_new_da_loss(src_latents, trgt_latents)

            # set total loss
            self.loss = pos_loss + mse_loss + self.da_loss_weight * da_loss

            # loss dictionary
            self.loss_dict["mse_loss"] = mse_loss.item()
            self.loss_dict["da_loss"] = da_loss.item()
            self.loss_dict["summed_loss"] = self.loss.item()

        def _my_new_da_loss(self, source_features, target_features):
            ...

Once registered, create a config for you UDA algorithm (see e.g. the example config for cmd (ADD LINK)). It could look something like this:

.. code-block:: yaml

    name: cmd
    da_loss_weight: 0.1
    kwargs:
        ...
        ...

To then use your new algorithm, simply link the respective config in the main.yaml (ADD LINK) config file and you are good to go!


Adding Model Selection Strategies
---------------------------------

To add an unsupervised model selection method, please register a function with `@register_model_selection_algorithm`. For an example, see the DEV implementation (ADD LINK).
Currently, the function can only take certain arguments that are computed in ``model_selector.py`` (ADD LINK). So if you need additional arguments, please modify this file and
pass them there.

Once created and registered, you can use your new model selection algorithm, by adding it to the arguments when running ``run_model_selection`` (ADD LINK).
