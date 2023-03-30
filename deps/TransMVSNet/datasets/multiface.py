from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from .data_io import *
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from pathlib import Path
import json
import matplotlib.pyplot as plt
import tqdm
from typing import Union


class MVSDataset(Dataset):
    """
    Dataset loader for Multiface Dataset
    """
    znear = 0.5
    zfar = 1.5

    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06,
                 split_config=Path("assets/data_splits/multiface/tiny_subset.json"),
                 downsample_factor=0.125,
                 **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = Path(datapath)
        self.split_config = split_config if isinstance(split_config, Path) else Path(split_config)
        self.mode = mode
        self.stage = self.mode
        self.nviews = nviews
        assert nviews == 4
        self.ndepths = ndepths
        self.kwargs = kwargs
        self.range_hor = 45
        self.range_vert = 30
        self.slide_range = 40
        self.downsample_factor = downsample_factor

        assert self.mode in ["train", "val", "test", "write_prediction"]
        self.metas = self.build_list()

    def build_list(self):

        meta_dir = Path("assets/data_splits/multiface")

        if self.mode in ["train", "write_prediction"]:
            stages = ["train"]
        elif self.mode in ["val", "test"]:
            stages = ["val"]
        elif self.mode == "all":
            stages = ["train", "val"]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # reading in diner metas
        diner_metas = list()
        for stage in stages:
            meta_fpath = meta_dir / (
                    stage + "_" + self.split_config.stem + ".txt")
            with open(meta_fpath, "r") as f:
                diner_metas_ = json.load(f)
            diner_metas += diner_metas_

        # converting diner metas to MaGNet-usable metas
        metas = list()
        processed_scans = []
        sample_idx = 0
        for meta in diner_metas:
            scan_identifier = str(Path(meta["scan_path"]).parents[1] / Path(meta["scan_path"]).name)
            if scan_identifier not in processed_scans:
                processed_scans.append(scan_identifier)
                assert self.nviews == len(meta["ref_ids"])
                for i in range(self.nviews):
                    ref_ids = [r for r in meta["ref_ids"][:i]] + \
                              [r for r in meta["ref_ids"][i + 1:]]
                    sample_meta = dict(idx=sample_idx,
                                       scan_path=meta["scan_path"],
                                       target_ids=meta["ref_ids"][i],  # (noptions,)
                                       ref_ids=ref_ids)  # (n_views-1, noptions)
                    metas.append(sample_meta)
                    sample_idx += 1
        return metas

    def __len__(self):
        return len(self.metas)

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
    def read_img(img_path):
        """
        returns H, W, 3 and H, W, 1
        """
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
        img = MVSDataset.gammaCorrect(img, dim=2).clip(0, 1).astype(np.float32)
        return img

    @staticmethod
    def read_alpha(p: Path):
        mask = np.array(Image.open(p)).astype(np.float32)[..., None] / 255.  # (H, W, 1)
        return mask

    def read_depth(self, dmap_path):
        gt_dmap = Image.open(dmap_path)
        gt_dmap = np.array(gt_dmap)[:, :, np.newaxis].astype(np.float32)  # (H, W, 1)
        gt_dmap *= 1e-4
        # gt_dmap = np.clip(gt_dmap, a_min=0, a_max=2.3)
        return gt_dmap

    def multiscale_x(self, x):
        h, w = x.shape
        x_ms = {
            "stage1": cv2.resize(x, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(x, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": x,
        }
        return x_ms

    @staticmethod
    def get_frame_n_subject(scan_path):
        frame, subject = scan_path.stem, scan_path.parents[-1].name
        return frame, subject

    @staticmethod
    def imgpath_to_dpath(p):
        return p.parents[3] / "depths" / p.relative_to(p.parents[2])

    @staticmethod
    def imgpath_to_apath(p):
        return p.parents[3] / "masks" / p.relative_to(p.parents[2])

    @staticmethod
    def int_2_viewname(i: int):
        return f"view_{i:05d}"

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

    def __getitem__(self, idx):
        meta = self.metas[idx]
        target_id = meta["target_ids"]
        ref_ids = np.array(meta["ref_ids"])

        scan_path = self.datapath / meta["scan_path"]
        subject = scan_path.parents[3].name
        seq = scan_path.parents[1].name
        frame = scan_path.stem

        # use only the reference view and first nviews-1 source views
        view_ids = [target_id] + ref_ids.tolist()

        cam_path = self.datapath / subject / "KRT"
        cam_dict = self.load_krt(cam_path)

        imgs = []
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_path = self.datapath / subject / "images" / seq / vid / (frame + ".png")
            extrinsics = np.concatenate((np.copy(cam_dict[vid]["extrin"]),
                                         np.array([[0., 0., 0., 1.]])), axis=0).astype(np.float32)
            extrinsics[:3, -1] /= 1000  # unit mm -> m
            intrinsics = np.copy(cam_dict[vid]["intrin"]).astype(np.float32)

            img = self.read_img(img_path)
            mask = self.read_alpha(self.imgpath_to_apath(img_path))
            H, W = img.shape[:2]
            h, w = int((H * self.downsample_factor) // 32 * 32), int((W * self.downsample_factor) // 32 * 32)  # ensure divisible by 32
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)[..., None]
            intrinsics[0] *= w / W
            intrinsics[1] *= h / H
            img[mask[..., 0] < 1] = 1.

            if i == 0:  # reference view
                dmap_path = self.imgpath_to_dpath(img_path)
                depth = self.read_depth(dmap_path)
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)[..., None]

                mask_ms = self.multiscale_x(mask[..., 0])
                depth_ms = self.multiscale_x(depth[..., 0])

                # get depth values
                depth_values = np.linspace(self.znear, self.zfar, self.ndepths, dtype=np.float32)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            imgs.append(img)

            # #####################################
            # # DEBUG: check depth map reprojection
            # #####################################
            # max_points = 100000000
            # outfile = f"/tmp/{vid}.txt"
            # K = torch.from_numpy(proj_mat[1][:3, :3])
            # Rt = torch.from_numpy(proj_mat[0])
            # K_inv = torch.linalg.inv(K)
            # Rt_inv = torch.linalg.inv(Rt)
            # dmap_path = self.imgpath_to_dpath(img_path)
            # depth = self.read_depth(dmap_path)
            # depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            # # create image rays
            # src_rays = torch.stack(torch.meshgrid(torch.arange(0.5, h, step=1.), torch.arange(0.5, w, step=1.))[::-1],
            #                        dim=-1)  # (h, w, 2)
            # src_rays = torch.cat((src_rays, torch.ones_like(src_rays[..., :1])), dim=-1)  # (h, w, 3)
            # src_rays = (K_inv @ src_rays.reshape(-1, 3).T).T  # (h*w, 3)
            # # projection into world space
            # src_points = src_rays * torch.from_numpy(depth).reshape(h * w)[..., None]  # (h * w, 3)
            # src_points = torch.cat((src_points, torch.ones_like(src_points[..., :1])), dim=-1)  # (h * w, 4)
            # world_points = (Rt_inv @ src_points.T).T  # (h * w, 4)
            # world_points = world_points[..., :3].reshape(-1, 3)  # (h*w, 3)
            # colors = torch.from_numpy(img).reshape(-1, 3)  # (h*w, 3)
            # idcs = np.random.permutation(np.arange(h * w))[:int(max_points)]
            # world_points = world_points[idcs]
            # colors = colors[idcs]
            # out = torch.cat((world_points, (colors * 255).round()), dim=-1).cpu().numpy()
            # np.savetxt(outfile, out, delimiter=";")
            # ##########################
            # # END DEBUG
            # ##########################

        # all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])  # B, 3, H, W

        # ms proj_mats
        H, W = imgs.shape[-2:]
        stage1_mults = (W // 4) / W, (H // 4) / H
        stage2_mults = (W // 2) / W, (H // 2) / H
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, 0, :] = proj_matrices[:, 1, 0, :] * stage1_mults[0]
        stage1_pjmats[:, 1, 1, :] = proj_matrices[:, 1, 1, :] * stage1_mults[1]
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, 0, :] = proj_matrices[:, 1, 0, :] * stage2_mults[0]
        stage2_pjmats[:, 1, 1, :] = proj_matrices[:, 1, 1, :] * stage2_mults[1]

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        return {"imgs": imgs,
                "dpath": str(dmap_path.relative_to(self.datapath)),
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "depth_interval": depth_values[1] - depth_values[0],
                "mask": mask_ms}

    def visualize_cam_grid(self, i=0, show=True):
        meta = self.metas[i]
        scan_path = self.datapath / meta["scan_path"]
        cam_path = scan_path / "cameras.json"
        with open(cam_path, "r") as f:
            cam_dict = json.load(f)
        cam_ids = sorted(cam_dict.keys())
        extrinsics = torch.from_numpy(np.stack([cam_dict[id]["extrinsics"] for id in cam_ids]))

        all_centers = -extrinsics[:, :3, :3].permute(0, 2, 1) @ extrinsics[:, :3, -1:]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        s = 50.
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                      s * extrinsics[:, i, 0], s * extrinsics[:, i, 1], s * extrinsics[:, i, 2],
                      edgecolor=color)
        for i, id in enumerate(range(49)):
            ax.text(all_centers[i, 0, 0], all_centers[i, 1, 0], all_centers[i, 2, 0], id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if show:
            plt.show()

    def visualize_item(self, i=0):
        sample = self.__getitem__(i)

        def print_dict_shape(d: dict):
            for key, val in d.items():
                if isinstance(val, dict):
                    print_dict_shape(val)
                else:
                    try:
                        print(key, val.shape)
                    except AttributeError:
                        print(key, val)

        print(print_dict_shape(sample))

        # visualizing rgb images
        imgs = np.concatenate([img for img in sample["imgs"]], axis=-1).transpose([1, 2, 0])
        plt.imshow(imgs)

        # visualizing multiscale dmaps and masks
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(9, 6))
        for j in range(3):
            dmap = sample["depth"][f"stage{j + 1}"]
            mask = sample["mask"][f"stage{j + 1}"]
            axes[0, j].imshow(dmap, vmin=dmap[dmap != 0].min())
            axes[1, j].imshow(mask)

        # visualize camera locations
        all_extrinsics = sample["proj_matrices"]["stage3"][:, 0]
        all_centers = -all_extrinsics[:, :3, :3].transpose([0, 2, 1]) @ all_extrinsics[:, :3, -1:]
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        s = .1
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                      s * all_extrinsics[:, i, 0], s * all_extrinsics[:, i, 1], s * all_extrinsics[:, i, 2],
                      edgecolor=color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_zlim((-1.5, 1.5))

        # checking reprojections
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3))
        random_points = np.random.normal(size=(10, 3)) * .1
        random_points[:, -1] += 1.
        random_points = np.concatenate((random_points, np.ones_like(random_points[:, :1])), axis=-1)
        for j in range(3):
            stage = f"stage{j + 1}"
            dmap = sample["depth"][stage]
            prj = sample["proj_matrices"][stage]
            prj_pts = (prj[0, 1] @ (prj[0, 0] @ random_points.T)).T
            prj_pts = prj_pts[..., :2] / prj_pts[..., 2:3]  # Npoints, 2 (u, v)
            axes[j].imshow(dmap, vmin=dmap[dmap != 0].min())
            axes[j].scatter(prj_pts[:, 0], prj_pts[:, 1], s=5., c="red")
        plt.show()

    def find_depth_range(self):
        dmin = 100000.
        dmax = 0.
        for i in tqdm.tqdm(np.random.permutation(np.arange(len(self)))):
            dmap = self.__getitem__(i)["depth"]["stage3"]
            dmin_ = dmap[dmap != 0].min()
            dmax_ = dmap.max()
            if dmin_ < dmin:
                print("new d_min:", dmin_)
                dmin = dmin_
            if dmax_ > dmax:
                print("new d_max:", dmax_)
                dmax = dmax_
        return dmin, dmax