import torch
from torch.nn.functional import pad
import matplotlib.pyplot as plt


@torch.no_grad()
def depth2normal(dmap, K):
    """
    calculating normal maps from depth map via central difference
    Parameters
    ----------
    dmap  (N, 1, H, W)
    K (N, 3, 3)

    Returns
    -------
    normal (N, 3, H, W)

    """
    N, _, H, W = dmap.shape
    device = dmap.device

    # reprojecting dmap to pointcloud
    image_rays = torch.stack(torch.meshgrid(torch.arange(0.5, H, 1., device=device),
                                            torch.arange(0.5, W, 1., device=device))[::-1],
                             dim=-1).reshape(-1, 2)  # H, W, 2
    image_rays = image_rays.unsqueeze(0).expand(N, -1, -1).clone()  # N, H*W, 2
    image_rays -= K[:, [0, 1], -1].unsqueeze(-2)
    image_rays /= K[:, [0, 1], [0, 1]].unsqueeze(-2)
    image_rays = torch.cat((image_rays, torch.ones_like(image_rays[..., -1:])), dim=-1)  # SB, H*W, 3
    image_pts = image_rays.view(N, H, W, 3) * dmap.view(N, H, W, 1)  # SB, H, W, 3
    image_pts = image_pts.permute(0, 3, 1, 2)  # SB, 3, H, W
    image_pts = pad(image_pts, [1] * 4, mode="replicate")  # SB, 3, H+2, W+2

    # # visualize pointcloud
    # fig = plt.figure()
    # ax=fig.add_subplot(projection="3d")
    # image_pts_ = image_pts[0].permute(1,2,0).view(-1, 3).cpu()
    # image_pts_ = image_pts_[image_pts_[:, 0]!=0]
    # ax.scatter(*image_pts_.unbind(dim=-1), s=5.)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.show()

    # calculating normals
    image_pts_offset_down = image_pts[:, :, 2:, 1:-1]  # SB, 3, H, W
    image_pts_offset_up = image_pts[:, :, :-2, 1:-1]  # SB, 3, H, W
    image_pts_offset_right = image_pts[:, :, 1:-1, 2:]  # SB, 3, H, W
    image_pts_offset_left = image_pts[:, :, 1:-1, :-2]  # SB, 3, H, W

    vdiff = image_pts_offset_down - image_pts_offset_up  # SB, 3, H, W
    hdiff = image_pts_offset_right - image_pts_offset_left  # SB, 3, H, W
    normal = torch.cross(vdiff.permute(0, 2, 3, 1), hdiff.permute(0, 2, 3, 1))  # SB, H, W, 3
    normal /= torch.norm(normal, p=2, dim=-1, keepdim=True)  # SB, H, W, 3

    # cleaning normal map
    idx_map = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(H), torch.arange(W)),
                          dim=-1).to(device)  # SB, H, W, 3
    offset_map = torch.zeros_like(idx_map)
    helper = (image_pts_offset_down[:, 0] == 0)[..., None] & \
             torch.tensor([False, True, False], device=device).view(1, 1, 1, 3)
    offset_map[helper] += -1
    helper = (image_pts_offset_up[:, 0] == 0)[..., None] & \
             torch.tensor([False, True, False], device=device).view(1, 1, 1, 3)
    offset_map[helper] += 1
    helper = (image_pts_offset_right[:, 0] == 0)[..., None] & \
             torch.tensor([False, False, True], device=device).view(1, 1, 1, 3)
    offset_map[helper] += -1
    helper = (image_pts_offset_left[:, 0] == 0)[..., None] & \
             torch.tensor([False, False, True], device=device).view(1, 1, 1, 3)
    offset_map[helper] += 1

    offset_mask = torch.any(offset_map != 0, dim=-1)
    new_idcs = idx_map[offset_mask] + offset_map[offset_mask]
    new_idcs[:, 1] = new_idcs[:, 1].clip(min=0, max=H - 1)
    new_idcs[:, 2] = new_idcs[:, 2].clip(min=0, max=W - 1)
    normal[offset_mask] = normal[new_idcs[:, 0], new_idcs[:, 1], new_idcs[:, 2]]
    normal[dmap[:, 0] == 0] = 0

    # # Visualize cleaned normal map
    # plt.imshow(normal[0].cpu() * .5 + .5)
    # plt.show()

    normal = normal.permute(0, 3, 1, 2)

    return normal
