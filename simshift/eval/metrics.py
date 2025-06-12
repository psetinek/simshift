import torch
import torch.nn.functional as F

from simshift.eval import get_metric, register_metric


@register_metric("mse_avg")
def mse_avg(pred, gt):
    return F.mse_loss(pred, gt, reduction="mean")


@register_metric("mse_max")
def mse_max(pred, gt):
    error = F.mse_loss(pred, gt, reduction="none")
    return torch.max(torch.mean(torch.flatten(error, start_dim=1), dim=-1))


@register_metric("ae_avg")
def ae_avg(pred, gt):
    error = torch.abs(pred - gt)
    return torch.mean(error)


@register_metric("ae_max")
def ae_max(pred, gt):
    error = torch.abs(pred - gt)
    return torch.max(torch.mean(torch.flatten(error, start_dim=1), dim=-1))


class Metrics:
    def __init__(self, metric_names=None):
        self.domains = ["source"]
        self.metric_names = metric_names or ["mse_avg", "ae_avg", "mse_max", "ae_max"]
        self.metric_funcs = {name: get_metric(name) for name in self.metric_names}
        self.reset_epoch()

    def __getitem__(self, key):
        if key not in self.metric_funcs:
            raise KeyError(
                f"Metric '{key}' not found. Available metrics: \
                    {list(self.metric_funcs.keys())}"
            )
        return self.metric_funcs[key]

    def __setitem__(self, key, value):
        if not callable(value):
            raise ValueError("Assigned metric must be callable.")
        self.metric_funcs[key] = value
        if key not in self.metric_names:
            self.metric_names.append(key)

    def calculate_all_metrics(self, pred, gt):
        metrics_dict = {}
        for name, fn in self.metric_funcs.items():
            value = fn(pred, gt)
            if "max" in name:
                metrics_dict[name] = value.item()
            else:
                bs = pred.shape[0]
                metrics_dict[name] = (
                    value.item() * bs,
                    bs,
                )  # also return batchsize for exact computation of mean with
                # incomplete batches
        return metrics_dict

    def reset_epoch(self):
        self._epoch_accum = {}
        for domain in self.domains:
            self._epoch_accum[domain] = {}
            for name in self.metric_names:
                if "max" in name:
                    self._epoch_accum[domain][name] = float("-inf")
                else:
                    self._epoch_accum[domain][name] = [
                        0.0,
                        0,
                    ]  # [weighted sum, total count]

    def update_domain_metrics(self, pred, gt, domain):
        batch_metrics = self.calculate_all_metrics(pred, gt)
        for name, value in batch_metrics.items():
            if "max" in name:
                self._epoch_accum[domain][name] = max(
                    self._epoch_accum[domain][name], value
                )
            else:
                current_sum, current_count = self._epoch_accum[domain][name]
                batch_sum, batch_count = value
                self._epoch_accum[domain][name] = [
                    current_sum + batch_sum,
                    current_count + batch_count,
                ]

    def get_epoch_stats(self):
        stats = {}
        for domain in self.domains:
            stats[domain] = {}
            for name in self.metric_names:
                if "max" in name:
                    stats[domain][name] = self._epoch_accum[domain][name]
                else:
                    total, count = self._epoch_accum[domain][name]
                    stats[domain][name] = total / count if count != 0 else None
        return stats

    def evaluate_domains(self, model, dataset, parameter_name, device="cuda"):
        parameters = list(
            dataset.cond_lookup.index.get_level_values(parameter_name).unique()
        )
        sample_counts = (
            dataset.cond_lookup.index.get_level_values(parameter_name)
            .value_counts()
            .values
        )
        losses = []
        model.eval()
        with torch.no_grad():
            for param in parameters:
                closest_value = (
                    dataset.cond_lookup.index.get_level_values(parameter_name)
                    .unique()
                    .to_series()
                    .sub(param)
                    .abs()
                    .idxmin()
                )

                # filter the df for the closest value of thickness and create batch
                resulting_df = dataset.cond_lookup.loc[
                    dataset.cond_lookup.index.get_level_values(parameter_name)
                    == closest_value
                ]
                filenames = resulting_df["filename"].to_list()
                conds = []
                ys = []
                for filename in filenames:
                    conds.append(dataset.data[filename]["cond"])
                    ys.append(dataset.data[filename]["y"])
                cond = torch.stack(conds)
                y = torch.stack(ys)

                # run model
                cond = cond.to(device)
                y = y.to(device)
                model = model.to(device)
                pred, _ = model(cond)
                # domain_loss = F.mse_loss(pred, y).item()
                domain_loss = torch.mean(self.absolute_error(pred, y)).item()
                losses.append(domain_loss)
        return parameters, losses, sample_counts


def get_metrics(cfg):
    return Metrics(metric_names=cfg.validation.metrics)
