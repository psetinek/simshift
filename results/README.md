## Benchmarking Experiments

To reproduce our results, we provide all necessary configuration files [here](../configs/). For each model we benchmark, the hyperparameters are listed in the [model configs](../configs/model/). Configurations for UDA algorithms can be found [here](../configs/da_algorithm/). We also provide the default dataset and training configs. Wandb is closely integrated in this repository. Please configure logging according to your needs [here](../configs/logging/wandb.yaml).

As an example, we now list all necessary steps to reproduce the results for PointNet combined with Deep-Coral on the rolling dataset. The other results can be reproduced analogously, by just specifying a different dataset, model or UDA algorithm.

The [main config file](../configs/main.yaml) is the entry point for hydra. From there you can link model, dataset and UDA algorithm cofigs, which are the more fine grained settings for the models, datasets or UDA algorithms respectively.

To start a run from the command line, adapt your config files as needed (for the example that we want to reproduce, just make sure that the rolling dataset, the PointNet model config and the deep_coral UDA algorithm config are linked) and simply run:

```
python main.py
```

Now for benchmarking purposes, one likely wants to spawn multiple runs at the same time. Therefore we provide a [launcher](../launcher.py) to make this easier. All arguments that are passed besides the max_concurrent processed and the gpus that should be used, will be sweeped over. So again to continue reproducing our partial result, we can simply run:

```
python launcher.py --max_concurrent=4 --gpus=0,1,2,3 seed=42,43,44,45 da_algorithm.da_loss_weight=0,0.1,0.01,0.001,0.0001,0.00001,1e-6,1e-7,1e-8,1e-9
```

This command will then train 4 models without domain adaptation regularization and a total 36 (9x4) models with the respective loss scaling terms. With this command, we will continuously spawn processes until all combinations of the provided parameters have been run.

After having trained those models, we will want to run unsupervised model selection. To run all implemented model selection strategies (Source Best (SB), Target Best(TB), Deep Embedded Validation (DEV) and Importance Weighted Validation (IWV)) for the runs above, this would look something like the following:

```
python run_model_selection.py --entity=<your_configured_wandb_entity> --project=<your_configured_wandb_project>  --experiment-id=<your_configured_wandb_experiment_id>  --model-dir=<your_configured_directoriy_of_ckpts> --batch-size=64 --device=cuda --output-dir=./results/model_selection_results --selection-algs SB TB DEV IWV
```

This script will save a dataframe containing the results reported in the main tables for PointNet + Deep Coral and all the different unsupervised model selection strategies. You can then postprocess them as you like to display your results, an example for postprocessing is given [here](./postprocess_results.ipynb).

Obtaining the results on other datasets and UDA algorithms is exactly analogous to the described steps, simply with e.g. a different model or dataset config linked in the [main config](../configs/main.yaml).
