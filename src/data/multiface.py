from torch.utils.data import random_split
from pathlib import Path
import os
import numpy as np
import torch
import json
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import time
from src.util.torch_helpers import dict_2_torchdict
from itertools import product
import tqdm
from src.util.cam_geometry import to_homogeneous_trafo
from typing import Union
import matplotlib.pyplot as plt
from src.util.cam_geometry import Slerp
from scipy.spatial.transform import Rotation

OPENCV2OPENGL = np.array([[1., 0., 0., 0.], [0., -1., 0., 0], [0., 0., -1., 0.], [0., 0., 0., 1.]], dtype=np.float32)


class MultiFaceDataset(torch.utils.data.Dataset):
    znear = 0.5
    zfar = 1.5

    def __init__(self, root: Path, stage, range_hor=45, range_vert=30, slide_range=0, slide_step=20.,
                 downsample=8, split_config=Path("assets/data_splits/multiface/tiny_subset.json"),
                 depth_suffix=".png", depth_std_suffix=None,
                 subject_filter=None, sequence_filter=None, target_filter=None,
                 manual_target_params=None):
        """
        MultiFace Data Loading Class.
        Camera extrinsics follow OPENCV convention (cams look in positive z direction, y points downwards)
        :param root:
        :param stage:
        :param source_views: if int: randomly samples <source_views> source views, if list of ints: samples these view ids
        :param banned_views:
        :param kwargs:
        """
        super().__init__()
        assert os.path.exists(root)
        self.data_dir = Path(root)
        self.stage = stage
        self.range_hor = range_hor  # inactive
        self.range_vert = range_vert  # inactive
        self.nsource = 4
        self.slide_range = slide_range  # inactive
        self.slide_step = slide_step  # inactive
        self.split_config = split_config if isinstance(split_config, Path) else Path(split_config)
        self.downsample = downsample
        self.depth_suffix = depth_suffix
        self.depth_std_suffix = depth_std_suffix
        assert isinstance(downsample, int)
        self.metas = self.get_metas(subject_filter=subject_filter, sequence_filter=sequence_filter,
                                    target_filter=target_filter)
        if manual_target_params is None:
            self.manual_target_params = None
        else:
            with open(manual_target_params, "r") as f:
                self.manual_target_params = json.load(f)
                assert len(self.manual_target_params["extrinsics"]) == self.__len__()

    @staticmethod
    def read_img(p: Path, symmetric_range=False, bg=1.):
        rgb = pil_to_tensor(Image.open(p)).float() / 255.  # range: 0 ... 1

        rgb = MultiFaceDataset.gammaCorrect(rgb, dim=0).clip(0, 1)

        if symmetric_range:  # rgb range: -1 ... 1
            rgb = rgb * 2 - 1

        return rgb

    @staticmethod
    def read_alpha(p: Path):
        mask = pil_to_tensor(Image.open(p)).float() / 255.  # 1, H, W
        return mask

    @staticmethod
    def gammaCorrect(img: Union[torch.Tensor, np.ndarray], dim: int = -3) -> Union[torch.Tensor, np.ndarray]:

        if dim < 0:
            dim += len(img.shape)
        assert (img.shape[dim] == 3)
        gamma, black, color_scale = 2.0, 3.0 / 255.0, [1.4, 1.1, 1.6]

        if dim == -1:
            dim = len(img.shape) - 1

        if torch.is_tensor(img):
            scale = torch.tensor(color_scale).view([3 if i == dim else 1 for i in range(img.dim())])
            img = img * scale.to(img) / 1.1
            return torch.clamp(
                (((1.0 / (1 - black)) * 0.95 * torch.clamp(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0, 0, 2, )
        else:
            scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
            img = img * scale / 1.1
            return np.clip((((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma)) - 15.0 / 255.0,
                           0, 2, )

    @staticmethod
    def read_depth(p: Path):
        UINT16_MAX = 65535
        SCALE_FACTOR = 1e-4

        img = pil_to_tensor(Image.open(p)).float() * SCALE_FACTOR  # convert depth to meters

        return img

    @staticmethod
    def load_krt(path):
        cameras = {}

        with open(path, "r") as f:
            while True:
                name = f.readline()
                if name == "":
                    break

                intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
                dist = [float(x) for x in f.readline().split()]
                extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
                f.readline()

                cameras[name[:-1]] = {
                    "intrin": np.array(intrin),
                    "dist": np.array(dist),
                    "extrin": np.array(extrin),
                }

        return cameras

    def get_metas(self, subject_filter=None, sequence_filter=None, target_filter=None):
        meta_dir = Path("assets/data_splits/multiface")
        meta_fpath = meta_dir / (
                self.stage + "_" + self.split_config.stem + ".txt")
        if meta_fpath.exists():
            with open(meta_fpath, "r") as f:
                metas = json.load(f)
        else:
            # creating metas
            print("creating metas")
            with open(self.split_config, "r") as f:
                split_config = json.load(f)
                split_config = split_config["train"] if self.stage == "train" else split_config["val"]

            metas = list()
            sample_idx = 0
            for subj in tqdm.tqdm(split_config["subjects"], desc="Subjects"):
                krt = self.load_krt(self.data_dir / subj / "KRT")
                cam_names = np.array(sorted(krt.keys()))
                extrinsics = np.array([krt[n]["extrin"] for n in cam_names])
                extrinsics = np.concatenate((extrinsics, np.zeros_like(extrinsics[:, :1])), axis=1)
                extrinsics[:, -1, -1] = 1
                cam_centers = (-extrinsics[:, :3, :3].transpose(0, 2, 1) @ extrinsics[:, :3, -1:])[..., 0]  # N x 3
                cam_dirs = extrinsics[:, 2, :3]  # N x 3

                origin = np.array([[0, 0, 1000.]])
                ideal_ref_centers = np.array(split_config["ref_centers"]).reshape(-1, 3)

                if subj == "m--20190529--1004--5067077--GHS":  # fixing error in dataset
                    rotation_y = lambda beta: np.array([[np.cos(beta), 0, np.sin(beta)],
                                                        [0, 1, 0],
                                                        [-np.sin(beta), 0, np.cos(beta)]])

                    ideal_ref_centers = (rotation_y(np.pi * 4 / 6) @ (ideal_ref_centers - origin).T).T + origin

                dists = np.sqrt(np.sum(((ideal_ref_centers[:, None] - cam_centers[None]) ** 2), axis=-1))  # (Nref, N)
                ref_idcs = np.argsort(dists, axis=1)[:, 0]
                ref_centers = cam_centers[ref_idcs]
                ref_dirs = cam_dirs[ref_idcs]
                ref_names = cam_names[ref_idcs].tolist()

                # only use target views inside spanned region
                plane_normals = np.cross(ref_centers[[0, 1, 2, 3]] - ref_centers[[1, 2, 3, 0]],
                                         ref_dirs[[0, 1, 2, 3]] + ref_dirs[[1, 2, 3, 0]])  # N_ref x 3
                plane_normals = plane_normals / np.sqrt(np.sum(plane_normals ** 2, axis=-1, keepdims=True))  # N_ref x 3
                inside_frustum_mask = np.sum((cam_centers[None] - ref_centers[:, None]) * plane_normals[:, None],
                                             axis=-1)
                inside_frustum_mask = np.all(inside_frustum_mask > -100, axis=0)  # max dist to plane: 10 cm
                inside_frustum_mask[ref_idcs] = False
                target_names = cam_names[inside_frustum_mask].tolist()

                # # visualize selection
                # import matplotlib.pyplot as plt
                # all_centers = -extrinsics[:, :3, :3].transpose(0, 2, 1) @ extrinsics[:, :3, -1:]
                #
                # fig = plt.figure()
                # ax = fig.add_subplot(projection="3d")
                # s = 100.
                # for i, color in enumerate(["red", "green", "blue"]):
                #     ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                #               s * extrinsics[:, i, 0], s * extrinsics[:, i, 1],
                #               s * extrinsics[:, i, 2],
                #               edgecolor=color)
                #
                # for i, id in enumerate(cam_names):
                #     ax.text(all_centers[i, 0, 0], all_centers[i, 1, 0], all_centers[i, 2, 0], id)
                #
                # ax.scatter(all_centers[ref_idcs][:, 0],
                #            all_centers[ref_idcs][:, 1],
                #            all_centers[ref_idcs][:, 2],
                #            c="black", zorder=0, s=60)
                # print(all_centers[ref_idcs])
                #
                # for i in range(len(ref_idcs)):
                #     ax.text(all_centers[ref_idcs][i, 0, 0],
                #            all_centers[ref_idcs][i, 1, 0],
                #            all_centers[ref_idcs][i, 2, 0], i)
                #
                # ax.scatter(all_centers[inside_frustum_mask][:, 0],
                #            all_centers[inside_frustum_mask][:, 1],
                #            all_centers[inside_frustum_mask][:, 2],
                #            c="orange", zorder=1, s=40)
                #
                # ax.set_xlabel("X")
                # ax.set_ylabel("Y")
                # ax.set_zlabel("Z")
                # plt.show()
                # plt.close()

                seq_paths = [p for p in sorted((self.data_dir / subj / "images").iterdir())
                             if p.name in split_config["sequences"]]
                for seq_path in tqdm.tqdm(seq_paths, desc="Sequences", leave=False):
                    for target_name in target_names:
                        target_img_dir = seq_path / target_name
                        frame_list = sorted(target_img_dir.iterdir())
                        for frame in frame_list:
                            sample_meta = dict(idx=sample_idx,
                                               scan_path=str(frame.relative_to(self.data_dir)),
                                               target_id=target_name,
                                               ref_ids=ref_names)
                            metas.append(sample_meta)
                            sample_idx += 1

            with open(meta_fpath, "w") as f:
                json.dump(metas, f, indent="\t")

        # Filtering metas
        if subject_filter is not None:
            metas = [m for m in metas if any([subj in m["scan_path"] for subj in subject_filter])]
        if sequence_filter is not None:
            metas = [m for m in metas if any([seq in m["scan_path"] for seq in sequence_filter])]
        if target_filter is not None:
            metas = [m for m in metas if any([targ == m["target_id"] for targ in target_filter])]

        return metas

    def __len__(self):
        return len(self.metas)

    @staticmethod
    def get_frame_n_subject(scan_path):
        frame, subject = scan_path.stem, scan_path.parents[-1].name
        return frame, subject

    def imgpath_to_dpath(self, p):
        return p.parents[3] / "depths" / p.relative_to(p.parents[2]).parent / (p.stem + self.depth_suffix)

    def imgpath_to_dstdpath(self, p):
        return p.parents[3] / "depths" / p.relative_to(p.parents[2]).parent / (p.stem + self.depth_std_suffix)

    @staticmethod
    def imgpath_to_apath(p):
        return p.parents[3] / "masks" / p.relative_to(p.parents[2])

    def __getitem__(self, idx):
        while True:  # working around random permission errors
            try:
                meta = self.metas[idx]

                # obtaining source view idcs
                source_ids = meta["ref_ids"]
                target_id = meta["target_id"]

                scan_path = Path(meta["scan_path"])
                subject = scan_path.parents[3].name
                seq = scan_path.parents[1].name
                frame = scan_path.stem

                target_img_path = self.data_dir / scan_path
                target_alpha_path = self.imgpath_to_apath(target_img_path)
                src_img_paths = [self.data_dir / subject / "images" / seq / source_id / (frame + ".png")
                                 for source_id in source_ids]
                cam_path = self.data_dir / subject / "KRT"

                target_depth_path = self.imgpath_to_dpath(target_img_path)
                src_depth_paths = [self.imgpath_to_dpath(p) for p in src_img_paths]
                src_depth_std_paths = [self.imgpath_to_dstdpath(p) for p in src_img_paths] \
                    if self.depth_std_suffix is not None else src_depth_paths

                target_rgb = self.read_img(target_img_path)
                target_alpha = self.read_alpha(target_alpha_path)
                src_rgbs = list()
                src_alphas = list()
                src_depths = list()
                src_depth_stds = list()
                for src_rgba_path, src_depth_path, src_depth_std_path in \
                        zip(src_img_paths, src_depth_paths, src_depth_std_paths):
                    src_rgb = self.read_img(src_rgba_path)

                    src_alpha = self.read_alpha(self.imgpath_to_apath(src_rgba_path))

                    src_depth = self.read_depth(src_depth_path)
                    if self.depth_std_suffix is None:
                        src_depth_std = torch.ones_like(src_depth) * 1e-3
                    else:
                        src_depth_std = self.read_depth(src_depth_std_path)
                        src_depth_std = (-1.582e-2 * src_depth_std + 1.649e-2).clip(min=0)
                    src_depth_std[src_depth == 0] = 0

                    src_rgbs.append(src_rgb), src_alphas.append(src_alpha), src_depths.append(src_depth)
                    src_depth_stds.append(src_depth_std)

                src_rgbs = torch.stack(src_rgbs)
                src_depths = torch.stack(src_depths)
                src_depth_stds = torch.stack(src_depth_stds)
                src_alphas = torch.stack(src_alphas)

                # white background
                src_rgbs.permute(0, 2, 3, 1)[src_alphas[:, 0] < 1] = 1
                target_rgb.permute(1, 2, 0)[target_alpha[0] < 1] = 1

                cam_dict = self.load_krt(cam_path)

                if self.manual_target_params is None:
                    target_extrinsics = torch.tensor(cam_dict[target_id]["extrin"]).float()
                    target_intrinsics = torch.tensor(cam_dict[target_id]["intrin"]).float()
                else:
                    target_extrinsics = torch.tensor(self.manual_target_params["extrinsics"][idx]).float()
                    target_intrinsics = torch.tensor(self.manual_target_params["intrinsics"][idx]).float()

                src_extrinsics = torch.tensor(np.array([cam_dict[src_id]["extrin"] for src_id in source_ids])).float()
                target_extrinsics = to_homogeneous_trafo(target_extrinsics[None]).float()[0]
                src_extrinsics = to_homogeneous_trafo(src_extrinsics).float()
                src_intrinsics = torch.tensor(np.array([cam_dict[src_id]["intrin"] for src_id in source_ids])).float()
                target_extrinsics[..., :3, -1] /= 1000  # unit mm -> meter
                src_extrinsics[..., :3, -1] /= 1000

                H, W = target_rgb.shape[-2:]
                h, w = int((H / self.downsample) // 32 * 32), int((W / self.downsample) // 32 * 32)

                if h != H or w != W:
                    # preprocess images so that exact downsampling through average pooling is possible
                    from torchvision.transforms.functional import resize, InterpolationMode
                    target_rgb = resize(target_rgb, (h, w))
                    src_rgbs = resize(src_rgbs, (h, w))
                    target_alpha = resize(target_alpha, (h, w), interpolation=InterpolationMode.NEAREST)
                    src_alphas = resize(src_alphas, (h, w), interpolation=InterpolationMode.NEAREST)

                    if src_depths.shape[-2:] != src_rgbs.shape[-2:]:
                        src_depths = resize(src_depths, (h, w), interpolation=InterpolationMode.NEAREST)
                        src_depth_stds = resize(src_depth_stds, (h, w), interpolation=InterpolationMode.NEAREST)

                    target_intrinsics[0] *= w / W
                    target_intrinsics[1] *= h / H
                    src_intrinsics[:, 0] *= w / W
                    src_intrinsics[:, 1] *= h / H

                sample = dict(target_rgb=target_rgb,
                              target_alpha=target_alpha,
                              target_extrinsics=target_extrinsics,
                              target_intrinsics=target_intrinsics,
                              target_view_id=torch.tensor(int(target_id)),
                              scan_idx=0,
                              sample_name=f"{subject}-{seq}-{frame}-{target_id}-{'-'.join(source_ids)}",
                              frame=frame,
                              src_rgbs=src_rgbs,
                              src_depths=src_depths,
                              src_depth_stds=src_depth_stds,
                              src_alphas=src_alphas,
                              src_extrinsics=src_extrinsics,
                              src_intrinsics=src_intrinsics,
                              src_view_ids=torch.tensor([int(src_id) for src_id in source_ids]))

                sample = dict_2_torchdict(sample)

                return sample
            except Exception as e:
                print("ERROR", e)
                time.sleep(np.random.uniform(.5, 1.))

    def get_cam_sweep_extrinsics(self, nframes, scan_idx, elevation=0., radius=1.8, sweep_range=None):
        base_sample = self.__getitem__(scan_idx)
        device = base_sample["target_extrinsics"].device

        src_extrinsics = base_sample["src_extrinsics"]
        src_pose = torch.linalg.inv(src_extrinsics)

        anchor_rots = Rotation.from_matrix(src_pose[:, :3, :3].cpu().numpy())
        anchor_rots = Rotation.concatenate((anchor_rots, anchor_rots[0], anchor_rots[2]))
        anchor_centers = src_pose[:, :3, -1]
        anchor_centers = torch.cat((anchor_centers,
                                    anchor_centers[0][None],
                                    anchor_centers[2][None]), dim=0).cpu().numpy()

        anchor_times = np.linspace(0, 1, len(anchor_rots))
        target_times = np.linspace(0, 1, nframes + 1)[:-1]
        slerp = Slerp(anchor_times, anchor_rots, anchor_centers)
        target_rotations, target_locations = slerp(target_times)

        target_poses = np.repeat(np.eye(4)[None], nframes, axis=0)
        target_poses[:, :3, :3] = target_rotations.as_matrix()
        target_poses[:, :3, -1] = target_locations
        target_poses = torch.from_numpy(target_poses).to(device)
        target_extrinsics = torch.linalg.inv(target_poses).float()

        # # visualize cam sweep poses
        # import matplotlib.pyplot as plt
        # target_poses = torch.linalg.inv(target_extrinsics)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # s = .1
        # for i, color in enumerate(["red", "green", "blue"]):
        #     ax.quiver(target_poses[:, 0, -1], target_poses[:, 1, -1], target_poses[:, 2, -1],
        #               s * target_poses[:, 0, i], s * target_poses[:, 1, i], s * target_poses[:, 2, i],
        #               edgecolor=color)
        # for i, id in enumerate(range(nframes)):
        #     ax.text(target_poses[i, 0, -1], target_poses[i, 1, -1], target_poses[i, 2, -1], str(id))
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.set_xlim(-1.5, 1.5)
        # ax.set_ylim(-1.5, 1.5)
        # ax.set_zlim(-1.5, 1.5)
        # plt.show()
        # plt.close()

        return target_extrinsics

    def visualize_item(self, idx):
        """
        plots item for debugging purposes
        :param idx:
        :return:
        """
        sample = self.__getitem__(idx)

        print(sample["target_view_id"], sample["src_view_ids"])

        # visualizing target and source views (rgb, alpha)
        import matplotlib.pyplot as plt
        ncols = self.nsource + 1
        nrows = 4
        s = 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(s * ncols, s * nrows))
        axes[0, 0].imshow(sample["target_rgb"].permute(1, 2, 0))
        axes[1, 0].imshow(sample["target_alpha"].permute(1, 2, 0))
        axes[0, 0].set_title(str(sample["target_view_id"]))
        for i in range(self.nsource):
            depth_masked = sample["src_depths"][i][sample["src_depths"][i] != 0]
            axes[0, i + 1].imshow(sample["src_rgbs"][i].permute(1, 2, 0))
            axes[1, i + 1].imshow(sample["src_alphas"][i].permute(1, 2, 0))
            axes[0, i + 1].set_title(str(sample["src_view_ids"][i]))
            axes[2, i + 1].imshow(sample["src_depths"][i].permute(1, 2, 0), vmin=depth_masked.min(),
                                  vmax=depth_masked.max())
            axes[3, i + 1].imshow(sample["src_depth_stds"][i].permute(1, 2, 0))
        [a.axis("off") for a in axes.flatten()]
        fig.suptitle(sample["sample_name"])
        plt.show()
        plt.close()

        # visualizing camera positions
        import matplotlib.pyplot as plt
        targ_extrinsics = sample["target_extrinsics"]
        src_extrinsics = sample["src_extrinsics"]
        all_extrinsics = torch.cat((targ_extrinsics.unsqueeze(0), src_extrinsics), dim=0)
        all_centers = -all_extrinsics[:, :3, :3].permute(0, 2, 1) @ all_extrinsics[:, :3, -1:]
        all_ids = [sample["target_view_id"].item()] + sample["src_view_ids"].tolist()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        s = .1
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                      s * all_extrinsics[:, i, 0], s * all_extrinsics[:, i, 1], s * all_extrinsics[:, i, 2],
                      edgecolor=color)
        for i, id in enumerate(all_ids):
            ax.text(all_centers[i, 0, 0], all_centers[i, 1, 0], all_centers[i, 2, 0], str(id))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_zlim((-1.5, 1.5))
        plt.show()
        plt.close()

    def reproject_depth(self, sample_idx=0, outfile=None, max_points=None):
        sample = self.__getitem__(sample_idx)
        src_imgs = sample["src_rgbs"]
        src_depths = sample["src_depths"]
        K = sample["src_intrinsics"]
        Rt = sample["src_extrinsics"]
        K_inv = torch.linalg.inv(K)
        Rt_inv = torch.linalg.inv(Rt)
        N = len(src_depths)

        # create image rays
        H, W = src_imgs.shape[-2:]
        src_rays = torch.stack(torch.meshgrid(torch.arange(0.5, H, step=1.), torch.arange(0.5, W, step=1.))[::-1],
                               dim=-1)  # (H, W, 2)
        src_rays = torch.cat((src_rays, torch.ones_like(src_rays[..., :1])), dim=-1)  # (H, W, 3)
        src_rays = src_rays[None].expand(N, -1, -1, -1)
        src_rays = (K_inv @ src_rays.reshape(N, -1, 3).permute(0, 2, 1)).permute(0, 2, 1)  # (N, H*W, 3)

        # projection into world space
        src_points = src_rays * src_depths[:, 0].reshape(N, H * W)[..., None]  # (N, H * W, 3)
        src_points = torch.cat((src_points, torch.ones_like(src_points[..., :1])), dim=-1)  # (N, H * W, 4)
        world_points = (Rt_inv @ src_points.permute(0, 2, 1)).permute(0, 2, 1)  # (N, H * W, 4)

        world_points = world_points[..., :3].reshape(-1, 3)  # (N*H*W, 3)
        colors = src_imgs.permute(0, 2, 3, 1).reshape(-1, 3)  # (N*H*W, 3)

        if max_points is not None:
            idcs = np.random.permutation(np.arange(N * H * W))[:int(max_points)]
            world_points = world_points[idcs]
            colors = colors[idcs]

        if outfile is not None:
            out = torch.cat((world_points, (colors * 255).round()), dim=-1).cpu().numpy()
            np.savetxt(outfile, out, delimiter=";")

        return world_points, colors

    def show_all_imgs(self, i=0):
        meta = self.metas[i]
        scan_path = self.data_dir / meta["scan_path"]
        print(scan_path)
        img_paths = [cam_path / scan_path.name for cam_path in sorted(scan_path.parents[1].iterdir())]
        n_cams = len(img_paths)
        n_grid = np.ceil(np.sqrt(n_cams)).astype(int)

        fig, axes = plt.subplots(ncols=n_grid, nrows=n_grid, figsize=(10, 10))
        for i, img_path in enumerate(img_paths):
            ax = axes.flatten()[i]
            ax.imshow(Image.open(img_paths[i]))
            ax.set_title(img_paths[i].parent.name)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def get_depth_range(self, n_samples=1000):
        dpaths = []
        for meta in np.random.choice(self.metas, replace=False, size=n_samples):
            scan_path = self.data_dir / meta["scan_path"]
            dpath = self.imgpath_to_dpath(scan_path)
            if dpath.exists():
                dpaths.append(dpath)

        d_min = 1000.
        d_max = 0.
        for dpath in tqdm.tqdm(dpaths, desc="Determining depth range"):
            dmap = self.read_depth(dpath)
            dmap_min = torch.min(dmap[dmap != 0]).item()
            dmap_max = torch.max(dmap).item()

            if dmap_min < d_min:
                print("New Min:", dmap_min)
                d_min = dmap_min
            if dmap_max > d_max:
                print("New Max:", dmap_max)
                d_max = dmap_max

        return d_min, d_max

    def visualize_camgrid(self, i=0):
        """
        plots item for debugging purposes
        :param idx:
        :return:
        """

        # visualizing camera positions
        import matplotlib.pyplot as plt
        meta = self.metas[i]
        scan_path = Path(meta["scan_path"])
        subject = scan_path.parents[3].name
        seq = scan_path.parents[1].name
        frame = scan_path.stem
        cam_path = self.data_dir / subject / "KRT"
        krt = self.load_krt(cam_path)
        cam_names = np.array(sorted(krt.keys()))
        extrinsics = np.array([krt[n]["extrin"] for n in cam_names])
        extrinsics = np.concatenate((extrinsics, np.zeros_like(extrinsics[:, :1])), axis=1)
        extrinsics[:, -1, -1] = 1
        cam_centers = (-extrinsics[:, :3, :3].transpose(0, 2, 1) @ extrinsics[:, :3, -1:])[..., 0]  # N x 3

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        s = 100.
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
                      s * extrinsics[:, i, 0], s * extrinsics[:, i, 1], s * extrinsics[:, i, 2],
                      edgecolor=color)
        for i, id in enumerate(cam_names):
            ax.text(cam_centers[i, 0], cam_centers[i, 1], cam_centers[i, 2], id)
        #
        # emph_cams = ["400300", "400349", "400347", "400365"]
        # mask = torch.zeros_like(torch.tensor(cam_centers[:, 0])).bool()
        # for name in emph_cams:
        #     mask = torch.logical_or(mask, torch.tensor(cam_names==name))
        # emph_cam_centers = cam_centers[mask]
        # emph_cam_viewdirs = extrinsics[mask][:, 2, :3]
        # ax.quiver(emph_cam_centers[:, 0], emph_cam_centers[:, 1], emph_cam_centers[:, 2],
        #           s * emph_cam_viewdirs[:, 0], s * emph_cam_viewdirs[:, 1], s * emph_cam_viewdirs[:, 2],
        #           edgecolor="pink")
        # ax.scatter(emph_cam_centers[:, 0], emph_cam_centers[:, 1], emph_cam_centers[:, 2], s=100, c="orange")
        #
        # view_dir_angles = np.arccos(np.clip(emph_cam_viewdirs@emph_cam_viewdirs.T, a_min=-0.9999, a_max=0.9999))*180/np.pi
        # print(emph_cam_centers)
        # print(krt)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        plt.close()
