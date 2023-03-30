import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
from torch import Tensor
from typing import Optional
from torch.nn.functional import pad, avg_pool2d, grid_sample as tgrid_sample
from torchvision.transforms.functional import resize

def dict_2_device(d: dict, device):
    d_out = dict()
    for key, val in d.items():
        if isinstance(val, torch.Tensor):
            val = val.to(device)
        d_out[key] = val
    return d_out


def dict_2_torchdict(d: dict):
    for key, val in d.items():
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val)
        elif isinstance(val, dict):
            val = dict_2_torchdict(val)

        d[key] = val
    return d


def unsqueeze_dict(d: dict):
    d_out = dict()
    for key, val in d.items():
        if isinstance(val, np.ndarray) or isinstance(val, torch.Tensor):
            val = val[None]
        if isinstance(val, str) or isinstance(val, int) or isinstance(val, float) or isinstance(val, list):
            val = [val]

        d_out[key] = val
    return d_out


def torch_cmap(x: torch.Tensor, cmap="viridis", vmin=None, vmax=None):
    """
    calculates a matplotlib colormap from tensor and returns result again as tensor

    Parameters
    ----------
    x (B, 1, H, W) or (1, H, W) or (H, W)

    Returns
    -------

    """

    cmap = plt.get_cmap(cmap)

    shape = x.shape
    device = x.device
    x = x.view(*[1 for i in range(4 - len(x.shape))], *x.shape)  # always bring x to shape (B, 1, H, W)
    assert x.shape[1] == 1

    x = x.detach().cpu().numpy().astype(float)
    vmin = vmin if vmin else np.min(x.reshape(x.shape[0], -1), axis=-1).reshape((-1, 1, 1, 1))
    vmax = vmax if vmax else np.max(x.reshape(x.shape[0], -1), axis=-1).reshape((-1, 1, 1, 1))
    x = (x - vmin) / (vmax - vmin)
    x = x[:, 0]  # (B, H, W)
    x = cmap(x)[..., :3]  # (B, H, W, 3)
    x = torch.from_numpy(x)
    x = x.permute(0, 3, 1, 2)  # (B, 3, H, W)

    outshape = list(shape[:-3]) + [3] + list(shape[-2:])
    x = x.reshape(outshape)
    x = x.to(device)

    return x


def save_torch_video(frames, outpath, fps):
    """

    Parameters
    ----------
    frames  (N, 3, H, W) with rgb values 0 ... 1
    outpath

    Returns
    -------

    """
    frames = (frames.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    while True:
        try:
            imageio.mimwrite(outpath, frames, fps=fps, quality=10)
            break
        except PermissionError:
            pass


def exponential_padding(img, padding: int, double_width):
    """
    implements padding strategy similar to border padding but exponentially increases border value with
    increasing distance to border.
    I.e: f(x) = f(x_border) * exp(ln(2) * (x-x_border) / double_width)
    Used for extrapolating depth standard deviation.
    :param img:
    :param padding:
    :param double_width:
    :return:
    """
    N, C, H, W = img.shape
    base = pad(img, [padding] * 4, mode="replicate")
    exponents = torch.zeros(*img.shape[:-2], H + 2 * padding, W + 2 * padding, dtype=img.dtype, device=img.device)
    for i in range(padding):
        idx = padding - (i + 1)
        exponents[:, :, idx, :] = i
        exponents[:, :, -(idx + 1), :] = i
        exponents[:, :, :, idx] = i
        exponents[:, :, :, -(idx + 1)] = i

    out = base * torch.exp(exponents / double_width * np.log(2))
    return out


def grid_sample(input: Tensor,
                grid: Tensor,
                mode: str = "bilinear",
                padding_mode: str = "zeros",
                align_corners: Optional[bool] = None,
                pad_double_width=20, pad_size=40, exp_padding_mode="border"
                ) -> Tensor:
    """
    extends pytorch grid sample by exponential padding
    Parameters
    ----------
    input
    grid
    mode
    padding_mode
    align_corners

    Returns
    -------

    """
    if padding_mode != "exponential":
        return tgrid_sample(input, grid, mode, padding_mode, align_corners)

    else:
        H, W = input.shape[-2:]
        img_size = torch.tensor([W, H], dtype=torch.float, device=input.device)
        input_padded = exponential_padding(input, pad_size, pad_double_width)

        # correcting grid
        if align_corners:  # -1 / +1 referring to outer pixel centers
            scale_factor = (img_size - 1) / (img_size + 2 * pad_size - 1)
        else:
            scale_factor = img_size / (img_size + 2 * pad_size)
        grid = grid * scale_factor.view(1, 1, 1, 2)
        return tgrid_sample(input_padded, grid, mode=mode, padding_mode=exp_padding_mode, align_corners=align_corners)


def masked_downsampling(x, mask, factor: int, mode="average", bg_color=0.):
    """
    allows for downsampling by integer factor of an image without colors of the background being washed into the fg.
    mask has no effect if mode=="nearest".
    Parameters
    ----------
    x
    mask
    factor: int
    mode
    bg_color

    Returns
    -------

    """
    device = x.device
    x_shape = x.shape
    assert x_shape[-1] % factor == 0
    assert x_shape[-2] % factor == 0

    if len(x_shape) == 3:
        x = x[None]  # B, C, H, W
        mask = mask[None]

    if mode == "average":
        x.permute(0, 2, 3, 1)[mask[:, 0] < 1] = 0
        x_sum = avg_pool2d(x, kernel_size=factor, stride=factor, divisor_override=1)
        mask_sum = avg_pool2d(mask, kernel_size=factor, stride=factor, divisor_override=1)
        mask_nearest = masked_downsampling(mask, mask, factor=factor, mode="nearest")
        fg_mask = mask_nearest[:, 0] > 0

        x_sum.permute(0, 2, 3, 1)[fg_mask] = x_sum.permute(0, 2, 3, 1)[fg_mask] / mask_sum.permute(0, 2, 3, 1)[fg_mask]
        x_sum.permute(0, 2, 3, 1)[mask_nearest[:, 0] == 0] = bg_color
        x_out = x_sum

    elif mode == "nearest":
        H, W = x_shape[-2:]
        vrange = torch.arange(factor / 2., H, factor, device=device) / H * 2 - 1
        urange = torch.arange(factor / 2., W, factor, device=device) / W * 2 - 1
        sample_grid = torch.stack(torch.meshgrid(vrange, urange)[::-1], dim=-1)
        sample_grid = sample_grid.expand(x.shape[0], -1, -1, -1)  # B, h, w, 2
        x_out = tgrid_sample(x, sample_grid, mode="nearest", padding_mode="border", align_corners=False)

    else:
        raise ValueError(f"Unreckognized mode '{mode}'")

    if len(x_shape) == 3:
        x_out = x_out.squeeze(0)

    return x_out


def weighted_mean_n_std(x: torch.Tensor, weights: torch.Tensor, dim: int, keepdims=False):
    weights_normed = weights / weights.sum(dim=dim, keepdims=True)
    mean = (x * weights_normed).sum(dim=dim, keepdims=True)
    std = ((x - mean).pow(2) * weights_normed).sum(dim=dim, keepdims=True).sqrt()

    if not keepdims:
        mean = mean.squeeze(dim)
        std = std.squeeze(dim)
    return mean, std
