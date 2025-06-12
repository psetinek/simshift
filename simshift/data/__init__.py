"""SIMSHIFT dataset factories and creation functions."""

import torch
from torch.utils.data import DataLoader

from .forming_data import get_forming_dataset
from .heatsink_data import get_heatsink_dataset
from .motor_data import get_motor_dataset
from .rolling_data import get_rolling_dataset

DATASET_BUILDERS = {
    "rolling": get_rolling_dataset,
    "forming": get_forming_dataset,
    "motor": get_motor_dataset,
    "heatsink": get_heatsink_dataset,
}


def get_data(cfg, val_only=False, normalization_stats=None):
    """Get a configured dataset depending on the passed OmegaConf config dictionary."""
    name = cfg.dataset.name
    dataset_kwargs = cfg.dataset
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers

    assert name in DATASET_BUILDERS

    datasets = []
    dataset_factory = DATASET_BUILDERS[name]

    if not val_only:
        trainset, normalization_stats = dataset_factory(split="train", **dataset_kwargs)
        trainset_source, trainset_target = trainset
    assert normalization_stats is not None

    valset, _ = dataset_factory(
        split="val", normalization_stats=normalization_stats, **dataset_kwargs
    )
    valset_source, valset_target = valset

    if not val_only:
        datasets += [trainset_source, valset_source, trainset_target, valset_target]
    else:
        datasets += [valset_source, valset_target]

    dataloaders = []
    # source dataloaders
    if not val_only:
        generator_source = torch.Generator().manual_seed(cfg.seed)
        trainloader_source = DataLoader(
            trainset_source,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=trainset_source.collate,
            generator=generator_source,
            drop_last=True,
        )
        dataloaders.append(trainloader_source)
    valloader_source = DataLoader(
        valset_source,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=valset_source.collate,
    )
    dataloaders.append(valloader_source)

    # target dataloaders
    if not val_only:
        generator_target = torch.Generator().manual_seed(
            cfg.seed + 1
        )  # different seed for target so sample pairs dont align for
        # da_loss computation
        trainloader_target = DataLoader(
            trainset_target,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=trainset_target.collate,
            generator=generator_target,
            drop_last=True,
        )
        dataloaders.append(trainloader_target)
    valloader_target = DataLoader(
        valset_target,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=valset_target.collate,
    )
    dataloaders.append(valloader_target)

    return datasets, dataloaders


def get_test_data(cfg, normalization_stats):
    """Get a configured test dataset based on the passed OmegaConf config dictionary."""
    name = cfg.dataset.name
    dataset_kwargs = cfg.dataset

    assert name in DATASET_BUILDERS
    dataset_factory = DATASET_BUILDERS[name]
    testset, _ = dataset_factory(
        split="test", normalization_stats=normalization_stats, **dataset_kwargs
    )

    return testset
