"""
Code heavily inspired by https://github.com/sxyu/pixel-nerf
"""

from torchvision.transforms import Normalize
import torch
from src.util.import_helper import import_obj
from src.models.positional_encoding import PositionalEncoding
from src.util.depth2normal import depth2normal


class PixelNeRF(torch.nn.Module):
    def __init__(self, poscode_conf, encoder_conf, mlp_fine_conf):
        super().__init__()
        self.poscode = PositionalEncoding(**poscode_conf.kwargs, d_in=3)
        self.depthcode = PositionalEncoding(**poscode_conf.kwargs, d_in=1)
        self.encoder = import_obj(encoder_conf.module)(**encoder_conf.kwargs)
        self.d_in = self.poscode.d_out + self.depthcode.d_out + 3
        self.d_latent = self.encoder.latent_size
        self.d_out = 4
        self.mlp_fine = import_obj(mlp_fine_conf.module)(**mlp_fine_conf.kwargs,
                                                         d_latent=self.d_latent,
                                                         d_in=self.d_in,
                                                         d_out=self.d_out)

        # Setting buffers for reference camera parameters
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        self.register_buffer("c", torch.empty(1, 2), persistent=False)

        self.normalize_rgb = Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def encode(self, images, depths, depths_std, extrinsics, intrinsics):
        """
        creates and stores feature maps, call encode() before using forward()!
        @param images: SB, NV, 3, H, W
        @param depths: SB, NV, 1, H, W
        @param extrinsics: SB, NV, 4, 4
        @param intrinsics: SB, NV, 3, 3
        @return:
        """
        images = self.normalize_rgb(images)
        normals = depth2normal(depths.flatten(end_dim=1), intrinsics.flatten(end_dim=1)).reshape_as(images)
        self.encoder(images, depths, depths_std, normals)
        self.poses = extrinsics
        self.c = intrinsics[:, :, :2, -1]
        self.focal = intrinsics[:, :, torch.tensor([0, 1]), torch.tensor([0, 1])]
        self.image_shape[0] = images.shape[-1]  # Width
        self.image_shape[1] = images.shape[-2]  # Height

        return

    def forward(self, xyz, viewdirs):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3)
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :param viewdirs (SB, B, 3)
        :return (SB, B, 4) r g b sigma
        """
        SB, B, _ = xyz.shape
        NV = self.encoder.nviews
        assert SB == self.encoder.nobjects

        # # visualize query points of first sample
        # import matplotlib.pyplot as plt
        # real_poses = torch.linalg.inv(self.poses).cpu()
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(*xyz[0, ::10].detach().cpu().unbind(dim=-1))
        # s=.1
        # for i, color in enumerate(["red", "green", "blue"]):
        #     ax.quiver(real_poses[0, :, 0, -1], real_poses[0, :, 1, -1], real_poses[0, :, 2, -1],
        #               s * real_poses[0, :, 0, i], s * real_poses[0, :, 1, i], s * real_poses[0, :, 2, i],
        #               edgecolor=color)
        # ax.scatter(*xyz[0, ::10].detach().cpu().unbind(dim=-1))
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_xlim((-1.5, 1.5))
        # ax.set_ylim((-1.5, 1.5))
        # ax.set_zlim((-1.5, 1.5))
        # plt.show()

        # Transform query points into the camera spaces of the input views
        xyz = xyz.unsqueeze(1).expand(-1, NV, -1, -1)  # (SB, NV, B, 3)
        xyz_rot = torch.matmul(self.poses[:, :, :3, :3], xyz.transpose(-2, -1)).transpose(-2, -1)
        xyz = xyz_rot + self.poses[:, :, :3, -1].unsqueeze(-2)  # (SB, NV, B, 3)

        # Positional encoding (no viewdirs)
        z_feature = self.poscode(xyz)  # SB, NV, B, d_pos

        # add viewdirs
        viewdirs = viewdirs.unsqueeze(1).expand(-1, NV, -1, -1)  # (SB, NV, B, 3)
        viewdirs = torch.matmul(self.poses[:, :, :3, :3],
                                viewdirs.transpose(-1, -2)).transpose(-1, -2)  # (SB, NV, B, 3)
        z_feature = torch.cat((z_feature, viewdirs), dim=-1)  # (SB, NV, B, d_in)

        # Grab encoder's latent code.
        uv = xyz[..., :2] / xyz[..., 2:]  # (SB, NV, B, 2)
        uv *= self.focal.unsqueeze(-2)
        uv += self.c.unsqueeze(-2)
        uv = uv / self.image_shape * 2 - 1  # assumes outer edges of pixels correspond to uv coordinates -1 / 1

        latent = self.encoder.index(uv)  # (SB, NV, latent, B)
        latent = latent.transpose(-1, -2)  # (SB, NV, B, latent)

        # encoding dist2depth
        ref_depth = self.encoder.index_depth(uv)  # SB, NV, 1, B
        depth_dist = ref_depth.squeeze(-2) - xyz[..., -1]  # (SB, NV, B)
        depth_feature = self.depthcode(depth_dist.unsqueeze(-1))  # (SB, NV, B, C)

        # # visualizing sampled points and mapped depths (*-1 and clipped to 1.)
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(xyz[0, 0, :, 0].detach().cpu(), xyz[0, 0, :, 1].detach().cpu(), xyz[0, 0, :, 2].detach().cpu(), s=.1)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # plt.show()

        mlp_input = torch.cat((latent, z_feature, depth_feature), dim=-1)  # (SB, NV, B, C_in)

        # Run main NeRF network
        mlp_output = self.mlp_fine(
            mlp_input,
            combine_dim=1
        )

        # Interpret the output
        mlp_output = mlp_output.reshape(SB, B, self.d_out)

        rgb = mlp_output[..., :3]
        sigma = mlp_output[..., 3:4]

        output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
        output = torch.cat(output_list, dim=-1)

        return output  # (SB, B, 4)