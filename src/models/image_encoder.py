"""
Code heavily inspired by https://github.com/sxyu/pixel-nerf
"""

import torch
from torch import nn
import torchvision
import functools
import torch.nn.functional as F
from src.util.torch_helpers import grid_sample
from src.models.positional_encoding import PositionalEncoding
import numpy as np

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
            self,
            backbone="resnet34",
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="border",
            upsample_interp="bilinear",
            use_first_pool=True,
            image_padding=0,
            padding_pe=-1
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        self.use_first_pool = use_first_pool
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained, norm_layer=norm_layer
        )
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
        self.image_padding = image_padding
        self.feature_padding = image_padding / self.model.conv1.stride[0]
        assert self.feature_padding % 1 == 0
        self.pad_layer = nn.ReplicationPad2d([self.image_padding] * 4)
        self.padding_pe = padding_pe

        if self.padding_pe < 0 or self.feature_padding == 0:
            self.positional_encoding = None
        else:
            self.positional_encoding = PositionalEncoding(padding_pe, freq_factor=np.pi, d_in=2, include_input=True)

            # adjusting input layer of pretrained resnet43 while copying pretrained remaining weights
            old_conv = self.model.conv1
            new_conv = torch.nn.Conv2d(old_conv.in_channels + self.positional_encoding.d_out,
                                       old_conv.out_channels,
                                       kernel_size=old_conv.kernel_size,
                                       stride=old_conv.stride,
                                       padding=old_conv.padding,
                                       bias=old_conv.bias,
                                       dilation=old_conv.dilation,
                                       padding_mode=old_conv.padding_mode,
                                       device=old_conv.weight.device,
                                       dtype=old_conv.weight.dtype,
                                       groups=old_conv.groups
                                       )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            new_conv.weight.requires_grad = False
            new_conv.weight[:, :old_conv.weight.shape[1]] = old_conv.weight.detach()
            new_conv.weight.requires_grad = True
            self.model.conv1 = new_conv

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        # self.latent (B, L, H, W)
        self.nviews = None
        self.nobjects = None

    def index(self, uv):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (SB, NV, N, 2) image points (x,y) in interval [-1, -1] (topleft pixelcorner)
                                                             to [1,1] (bottomright pixelcorner)
        :return (SB, NV, L, N) L is latent size
        """

        assert uv.shape[:2] == self.latent.shape[:2]

        SB, NV, N, _ = uv.shape
        N_ = SB * NV
        uv = uv.view(N_, N, 2)
        latent = self.latent.view(N_, *self.latent.shape[-3:])

        # correcting uv for feature padding
        latent_size = torch.tensor([self.latent.shape[-1], self.latent.shape[-2]], device=uv.device)  # W+pad, H+pad
        uv = uv * ((latent_size - self.feature_padding * 2) / latent_size).view(1, 1, 2)

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = F.grid_sample(
            latent,
            uv,
            align_corners=False,
            mode=self.index_interp,
            padding_mode=self.index_padding,
        )
        samples = samples[:, :, :, 0]  # (N_, C, N)

        # # printing sample values of first layer (64dims) on center 100x100px on fg, use mode="nearest"
        # # and set self.model.bn1 = nn.Sequential()
        # uv_mask = torch.all(uv.abs() < 100. / latent.shape[-1], dim=-1)[:, :, 0]
        # samples_100 = samples.permute(0, 2, 1)[uv_mask][..., :64]
        # if self.image_padding != 0:
        #     torch.save(samples_100, "/tmp/test.pt")
        # else:
        #     print(torch.all(samples_100==torch.load("/tmp/test.pt")))
        #
        # # visualize sampling
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # plt.imshow(latent[0, :3].permute(1, 2, 0).detach().cpu())
        # plt.scatter(*((uv[0, :, 0].cpu() + 1.) * latent.shape[-1] / 2 - .5).unbind(dim=-1), s=10.)
        # plt.xlim((-1, latent.shape[-1]))
        # plt.ylim((latent.shape[-1], -1))
        # if self.feature_padding == 0:
        #     plt.show()

        samples = samples.view(SB, NV, *samples.shape[-2:])  # (SB, NV, C, N)
        return samples

    def index_depth(self, uv):
        """
        Get pixel-aligned depths at 2D image coordinates (copied from self.index())
        :param uv (SB, NV, N, 2) image points (x,y)
        :return (SB, NV, 1, N)
        """
        assert uv.shape[:2] == self.depths.shape[:2]
        SB, NV, N, _ = uv.shape
        N_ = SB * NV
        uv = uv.view(N_, N, 2)
        depths = self.depths.view(N_, *self.depths.shape[-3:])

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = F.grid_sample(
            depths,
            uv,
            align_corners=False,
            mode="nearest",
            padding_mode="border",
        )
        samples = samples[:, :, :, 0]  # (N_, C, N)
        samples = samples.view(SB, NV, *samples.shape[-2:])  # (SB, NV, C, N)
        return samples

    def index_depth_std(self, uv):
        """
        Get pixel-aligned depth standard deviations at 2D image coordinates (copied from self.index())
        :param uv (SB, NV, N, 2) image points (x,y)
        :return (SB, NV, 1, N)
        """
        assert uv.shape[:2] == self.depths_std.shape[:2]
        SB, NV, N, _ = uv.shape
        N_ = SB * NV
        uv = uv.view(N_, N, 2)
        depths_std = self.depths_std.view(N_, *self.depths_std.shape[-3:])

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)

        samples = grid_sample(
            depths_std,
            uv,
            align_corners=False,
            mode="nearest",
            padding_mode="exponential",
            pad_double_width=12,
            pad_size=100,
            exp_padding_mode="zeros"
        )

        samples = samples[:, :, :, 0]  # (N_, C, N)
        samples = samples.view(SB, NV, *samples.shape[-2:])  # (SB, NV, C, N)
        return samples

    def index_normal(self, uv):
        """
        Get pixel-aligned normals at 2D image coordinates (copied from self.index())
        :param uv (SB, NV, N, 2) image points (x,y)
        :return (SB, NV, 1, N)
        """
        assert uv.shape[:2] == self.normals.shape[:2]
        SB, NV, N, _ = uv.shape
        N_ = SB * NV
        uv = uv.view(N_, N, 2)
        normals = self.normals.view(N_, *self.normals.shape[-3:])

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        samples = F.grid_sample(
            normals,
            uv,
            align_corners=False,
            mode="nearest",
            padding_mode="zeros",
        )
        samples = samples[:, :, :, 0]  # (N_, C, N)
        samples = samples.view(SB, NV, *samples.shape[-2:])  # (SB, NV, C, N)
        return samples

    def forward(self, imgs, depths, depths_std, normals):
        """
        For extracting ResNet's features and storing depth maps. Call before using self.index()
        :param imgs image (SB, NV, C, H, W)
        :return latent (SB, NV, latent_size, H, W)
        """

        SB, NV, Cin, H, W = imgs.shape
        self.depths = depths
        self.depths_std = depths_std
        self.normals = normals
        self.nviews = NV
        self.nobjects = SB

        # creating latent feature maps
        N_ = SB * NV
        imgs = imgs.view(N_, Cin, H, W)  # flattening along batch dimensions for feature extraction

        imgs = self.pad_layer(imgs)

        if self.padding_pe >= 0 and self.feature_padding > 0:
            pe_in = torch.stack(
                torch.meshgrid(torch.linspace(-1, 1, H + 2 * self.image_padding, device=imgs.device),
                               torch.linspace(-1, 1, W + 2 * self.image_padding, device=imgs.device))[::-1], dim=-1)
            pe_in = self.positional_encoding(pe_in)
            pe_in[self.image_padding:-self.image_padding, self.image_padding:-self.image_padding] = 0
            imgs = torch.cat((imgs, pe_in.permute(2, 0, 1).unsqueeze(0).expand(N_, -1, -1, -1)), dim=1)

            # # visualizing positional encoding
            # import matplotlib.pyplot as plt
            # try:
            #     plt.imshow(imgs[0, -1, :, :].detach().cpu())
            #     plt.show()
            # except Exception:
            #     pass

        imgs = self.model.conv1(imgs)
        imgs = self.model.bn1(imgs)
        imgs = self.model.relu(imgs)

        latents = [imgs]
        if self.num_layers > 1:
            if self.use_first_pool:
                imgs = self.model.maxpool(imgs)
            imgs = self.model.layer1(imgs)
            latents.append(imgs)
        if self.num_layers > 2:
            imgs = self.model.layer2(imgs)
            latents.append(imgs)
        if self.num_layers > 3:
            imgs = self.model.layer3(imgs)
            latents.append(imgs)
        if self.num_layers > 4:
            imgs = self.model.layer4(imgs)
            latents.append(imgs)

        align_corners = None if self.index_interp == "nearest " else True
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode=self.upsample_interp,
                align_corners=align_corners,
            )
        self.latent = torch.cat(latents, dim=1)
        self.latent = self.latent.view(SB, NV, -1, *self.latent.shape[-2:])

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )
