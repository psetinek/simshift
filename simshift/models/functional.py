from torch import nn

ALL_CONVS = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)

ALL_LAYERS = (
    nn.Linear,
    *ALL_CONVS,
)


def init_truncnormal_zero_bias(m: nn.Module, std: float = 0.02) -> None:
    if isinstance(m, ALL_LAYERS):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
