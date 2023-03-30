"""
Code heavily inspired by https://github.com/sxyu/pixel-nerf
"""

import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (D1, D2, ..., self.d_in)
        :return (D1, D2, ..., self.d_out)
        """

        # flattening x if it has leading batch dimensions
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], -1)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)

        if embed.shape[:-1] != original_shape[:-1]:
            embed = embed.reshape(*original_shape[:-1], self.d_out)
        return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )


if __name__ == "__main__":
    from torchvision.utils import save_image
    H, W = 256, 256
    pad = 200
    outpath = "/tmp/test.png"

    # plotting positional encoding
    pe = torch.stack(
        torch.meshgrid(torch.linspace(-1, 1, H + 2 * pad),
                       torch.linspace(-1, 1, W + 2 * pad))[::-1], dim=-1)
    pe = PositionalEncoding(d_in=2, num_freqs=4)(pe)
    pe = torch.stack([t.sum(dim=-1) for t in torch.tensor_split(pe, 3, dim=-1)])
    pe_min = torch.min(pe.view(3, -1), dim=-1).values.view(3, 1, 1)
    pe_max = torch.max(pe.view(3, -1), dim=-1).values.view(3, 1, 1)
    pe = (pe-pe_min)/(pe_max-pe_min)
    pe[:, pad:-pad, pad:-pad] = 0.5
    save_image(pe, outpath)