from torch.utils.data import random_split
from pathlib import Path
import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from src.util.cam_geometry import get_ray_intersections
from src.util.torch_helpers import dict_2_torchdict
from itertools import product
from src.util.io import read_pfm
from scipy.spatial.transform import Rotation, Slerp
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import tqdm


class DTUDataSet(torch.utils.data.Dataset):

    def __init__(self, root: Path, stage,
                 scale_factor=0.7 / 872.,
                 downsample=.5,
                 depth_fname="TransMVSNet"):
        """
        DTU Data Loading Class.
        Camera extrinsics follow OPENCV convention (cams look in positive z direction, y points down)
        :param root:
        :param stage: "train" or "val"
        :param scale_factor: 3d space scaling factor between DTU and FaceScape
        :param downsample: image downsampling factor
        """
        super().__init__()
        assert os.path.exists(root)
        self.data_dir = Path(root)
        self.stage = stage
        self.scale_factor = scale_factor
        self.downsample = downsample
        self.depth_fname = depth_fname

        self.scan_list = self.get_scan_list()
        self.cam_dict = self.get_cam_dict()
        self.znear = 400 * self.scale_factor
        self.zfar = 1500 * self.scale_factor

        self.nscans = len(self.scan_list)
        self.ncams = len(self.cam_dict["ids"])
        self.nlights = 7
        self.src_camids = [30, 10, 6, 35]
        self.conf2std = self._getconf2std()

        self.metas = self.get_metas()

    def get_metas(self):
        """
        get list of meta information for dataset samples
        :return: list with entries being dicts with keys: scan_idx, cam_idx, ref_cam_idcs, light_idx
        """
        metas = list()
        for scan_idx, cam_idx, light_idx in \
                product(range(self.nscans), range(self.ncams), range(self.nlights)):
            metas.append(dict(scan_idx=scan_idx, cam_idx=cam_idx, ref_cam_idcs=self.src_camids, light_idx=light_idx))
        return metas

    @staticmethod
    def camname2int(s: str):
        return int(s.strip("_cam.txt"))

    def _getconf2std(self):
        conf2std = lambda x: -2.5679e-2 * x + 3.2818e-2
        return conf2std

    def read_rgb(self, p: Path, symmetric_range=False):
        """
        loads image from as normalized torch tensor
        :param p:
        :param symmetric_range: if true: values range from -1 ... +1 else from 0 to 1
        :return:
        """
        rgb = Image.open(p)
        if self.downsample:
            w, h = rgb.size
            rgb = rgb.resize((int(w * self.downsample), int(h * self.downsample)))
        rgb = pil_to_tensor(rgb).float() / 255.  # range: 0 ... 1

        if symmetric_range:  # rgb range: -1 ... 1
            rgb = rgb * 2 - 1

        return rgb

    def read_depth(self, filename):
        """
        reading depth from either .png or .pfm file and returns it as torch tensor
        :param filename:
        :return:
        """
        if str(filename).endswith(".pfm"):
            depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
            depth_h = torch.from_numpy(depth_h)
            H, W = depth_h.shape
            h, w = int(H / 2), int(W / 2)
            depth_h = resize(depth_h[None][None], (h, w), interpolation=InterpolationMode.NEAREST)[0, 0]
            depth_h = depth_h[44:556, 80:720]  # (512, 640)

        elif str(filename).endswith(".png"):  # loading TransMVSNet prediction
            SCALE_FACTOR = 1e-4
            depth_h = pil_to_tensor(Image.open(filename)).float() * SCALE_FACTOR
            depth_h /= (0.7 / 872.)  # have to correct for scale factor used during TransMVSNet training
            depth_h = depth_h[0]  # (512, 640)

        else:
            raise ValueError

        h, w = depth_h.shape[:2]
        assert h == 512 and w == 640
        if self.downsample != 1:
            h, w = int(h * self.downsample), int(w * self.downsample)
            depth_h = resize(depth_h[None][None], (h, w), interpolation=InterpolationMode.NEAREST)[0, 0]
        mask = (depth_h > 0).float()
        depth_h *= self.scale_factor

        depth_h = depth_h[None]  # (1, H, W)
        mask = mask[None]  # (1, H, W)

        return depth_h, mask

    @staticmethod
    def int_to_viewdir(i: int):
        return f"view_{i:05d}"

    def get_scan_list(self):
        # obtaining scan lists
        if self.stage == "train":
            scan_list_file = "assets/data_splits/dtu/dtu_train_all.txt"
        elif self.stage == "val":
            scan_list_file = "assets/data_splits/dtu/dtu_val_all.txt"
        else:
            raise ValueError

        scan_list = np.loadtxt(scan_list_file, str)
        return scan_list

    @staticmethod
    def read_cam_file(filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = depth_min + float(lines[11].split()[1]) * 192
        return intrinsics, extrinsics, [depth_min, depth_max]

    def get_cam_dict(self):
        # loading cameras
        camera_dir = self.data_dir / "Cameras/train"
        cam_extrinsics = list()
        cam_intrinsics = list()

        cam_paths = [f for f in sorted(camera_dir.iterdir())
                     if f.name.endswith("_cam.txt")]
        cam_ids = [self.camname2int(f.name) for f in cam_paths]
        for cam_path in cam_paths:
            intrinsics, extrinsics, (depth_min, depth_max) = self.read_cam_file(cam_path)
            intrinsics[:2] *= 4
            intrinsics[:2] = intrinsics[:2] * self.downsample
            extrinsics[:3, 3] *= self.scale_factor
            depth_min *= self.scale_factor
            depth_max *= self.scale_factor
            cam_extrinsics.append(extrinsics)
            cam_intrinsics.append(intrinsics)
        cam_ids = torch.tensor(cam_ids)
        cam_extrinsics = torch.from_numpy(np.stack(cam_extrinsics))
        cam_intrinsics = torch.from_numpy(np.stack(cam_intrinsics))
        cam_dict = dict(ids=cam_ids, extrinsics=cam_extrinsics, intrinsics=cam_intrinsics)

        return cam_dict

    def __len__(self):
        return len(self.metas)

    def get_depth_fname(self, cam_id):
        name = f"depth_map_{cam_id:04d}_{self.depth_fname}.png"
        return name

    def __getitem__(self, idx):
        meta_dict = self.metas[idx]
        scan_idx = meta_dict["scan_idx"]
        cam_idx = meta_dict["cam_idx"]
        ref_cam_idcs = meta_dict["ref_cam_idcs"]
        light_idx = meta_dict["light_idx"]
        scan_name = self.scan_list[scan_idx]

        all_cam_idcs = [cam_idx] + ref_cam_idcs
        all_cam_ids = [self.cam_dict["ids"][i] for i in all_cam_idcs]

        all_img_paths = [self.data_dir / "Rectified" / (scan_name + "_train") /
                         f"rect_{i + 1:03d}_{light_idx}_r5000.png"
                         for i in all_cam_ids]

        all_depth_paths = [self.data_dir / "Depths" / scan_name / self.get_depth_fname(i)
                           for i in all_cam_ids[1:]]

        all_imgs = [self.read_rgb(p) for p in all_img_paths]
        all_depths, all_masks = zip(*[self.read_depth(p) for p in all_depth_paths])

        all_intrinsics = [self.cam_dict["intrinsics"][i] for i in all_cam_idcs]
        all_extrinsics = [self.cam_dict["extrinsics"][i] for i in all_cam_idcs]

        all_imgs = torch.stack(all_imgs)
        all_masks = torch.stack(all_masks)
        all_depths = torch.stack(all_depths)
        all_intrinsics = torch.stack(all_intrinsics)
        all_extrinsics = torch.stack(all_extrinsics)
        all_cam_ids = torch.tensor(all_cam_ids)

        all_depth_std_paths = [p.parent / p.name.replace(".png", "_conf.png") for p in all_depth_paths]
        all_depth_stds = [self.read_depth(p)[0] for p in all_depth_std_paths]
        all_depth_stds = torch.stack(all_depth_stds)
        all_depth_stds = self.conf2std(all_depth_stds)

        sample = dict(target_rgb=all_imgs[0],
                      target_alpha=torch.ones_like(all_imgs[0, :1]),
                      target_extrinsics=all_extrinsics[0],
                      target_intrinsics=all_intrinsics[0],
                      target_view_id=all_cam_ids[0],
                      scan_idx=scan_idx,
                      sample_name=f"{scan_name}-{all_cam_ids[0]}",
                      src_rgbs=all_imgs[1:],
                      src_alphas=all_masks,
                      src_depths=all_depths,
                      src_depth_stds=all_depth_stds,
                      src_extrinsics=all_extrinsics[1:],
                      src_intrinsics=all_intrinsics[1:],
                      src_view_ids=all_cam_ids[1:],
                      light_idx=light_idx)

        sample = dict_2_torchdict(sample)

        return sample

    @torch.no_grad()
    def get_cam_sweep_extrinsics(self, nframes, scan_idx=None, elevation=0., radius=.5):
        """

        :param nframes:
        :param scan_idx: no effect but there for compability reasons
        :param elevation: no effect but there for compability reasons
        :param radius: no effect but there for compability reasons
        :return:
        """

        center_extr = self.cam_dict["extrinsics"][24]
        left_extr = self.cam_dict["extrinsics"][11]
        right_extr = self.cam_dict["extrinsics"][18]

        center_pose = torch.linalg.inv(center_extr)
        left_pose = torch.linalg.inv(left_extr)
        right_pose = torch.linalg.inv(right_extr)

        center_camray = torch.cat((center_pose[:3, -1], center_pose[:3, -2]))
        left_camray = torch.cat((left_pose[:3, -1], left_pose[:3, -2]))
        right_camray = torch.cat((right_pose[:3, -1], right_pose[:3, -2]))

        rotation_origin = torch.mean(
            torch.stack(
                get_ray_intersections(left_camray, center_camray)
                + get_ray_intersections(center_camray, right_camray)
                + get_ray_intersections(left_camray, right_camray), dim=0
            ), dim=0
        )
        radius = (torch.norm(rotation_origin - left_pose[:3, -1], p=2)
                  + torch.norm(rotation_origin - center_pose[:3, -1], p=2)
                  + torch.norm(rotation_origin - right_pose[:3, -1], p=2)) / 3

        # pose interpolation
        t = torch.linspace(0, 1, nframes)

        # interpolate camera centers
        # see https://en.wikipedia.org/wiki/Slerp
        x1 = left_pose[:3, -1] - rotation_origin
        x2 = center_pose[:3, -1] - rotation_origin
        x3 = right_pose[:3, -1] - rotation_origin
        x1 /= torch.norm(x1, p=2)
        x2 /= torch.norm(x2, p=2)
        x3 /= torch.norm(x3, p=2)
        theta1 = torch.acos(torch.matmul(x1, x2).clip(min=-1, max=1.))
        theta2 = torch.acos(torch.matmul(x2, x3).clip(min=-1, max=1.))
        x1 = x1[None]
        x2 = x2[None]
        x3 = x3[None]
        target_centers = torch.zeros(nframes, 3, dtype=torch.float, device=x1.device)
        first_half_mask = t < .5
        t1 = t[first_half_mask] * 2
        t2 = t[~first_half_mask] * 2 - 1
        target_centers[first_half_mask] = (torch.sin((1 - t1[:, None]) * theta1) / torch.sin(theta1) * x1
                                           + torch.sin(t1[:, None] * theta1) / torch.sin(theta1) * x2)
        target_centers[~first_half_mask] = (torch.sin((1 - t2[:, None]) * theta2) / torch.sin(theta2) * x2
                                            + torch.sin(t2[:, None] * theta2) / torch.sin(theta2) * x3)
        target_centers *= radius
        target_centers += rotation_origin[None]

        # interpolating camera rotations
        rot1 = Rotation.from_matrix(left_pose[:3, :3])
        rot2 = Rotation.from_matrix(center_pose[:3, :3])
        rot3 = Rotation.from_matrix(right_pose[:3, :3])
        slerp = Slerp([0., 0.5, 1.], Rotation.concatenate([rot1, rot2, rot3]))
        target_rots = torch.tensor(slerp(t.numpy()).as_matrix())

        target_poses = torch.eye(4)[None].repeat(nframes, 1, 1)
        target_poses[:, :3, :3] = target_rots
        target_poses[:, :3, -1] = target_centers
        target_extrinsics = torch.linalg.inv(target_poses)

        # # visualize cam sweep extrinsics
        # import matplotlib.pyplot as plt
        # all_extrinsics = torch.cat((left_extr[None], center_extr[None], right_extr[None],
        #                             target_extrinsics), dim=0)
        # all_ids = ["left", "center", "right"]
        # all_centers = -all_extrinsics[:, :3, :3].permute(0, 2, 1) @ all_extrinsics[:, :3, -1:]
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # s = .1
        # for i, color in enumerate(["red", "green", "blue"]):
        #     ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
        #               s * all_extrinsics[:, i, 0], s * all_extrinsics[:, i, 1], s * all_extrinsics[:, i, 2],
        #               edgecolor=color)
        # for i, id in enumerate(all_ids):
        #     ax.text(all_centers[i, 0, 0], all_centers[i, 1, 0], all_centers[i, 2, 0], str(id))
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
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
        sample_name = f"{sample['sample_name']}-{str(sample['src_view_ids'])}-{sample['light_idx']}"

        # visualizing target and source views (rgb, alpha and depth)
        import matplotlib.pyplot as plt
        ncols = len(self.src_camids) + 1
        nrows = 3
        s = 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(s * ncols, s * nrows))
        axes[0, -1].imshow(sample["target_rgb"].permute(1, 2, 0))
        axes[0, -1].set_title(sample["target_view_id"])
        for i in range(len(self.src_camids)):
            depth_masked = sample["src_depths"][i][sample["src_depths"][i] < 3.]
            axes[0, i].imshow(sample["src_rgbs"][i].permute(1, 2, 0))
            axes[1, i].imshow(sample["src_alphas"][i].permute(1, 2, 0))
            axes[2, i].imshow(sample["src_depths"][i].permute(1, 2, 0), vmin=depth_masked.min(),
                              vmax=depth_masked.max())
            axes[0, i].set_title(str(sample["src_view_ids"][i]))
        fig.suptitle(sample_name)
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
        plt.show()
        plt.close()

    def visualize_camgrid(self):
        """
        plots item for debugging purposes
        :param idx:
        :return:
        """

        # visualizing camera positions
        import matplotlib.pyplot as plt
        all_extrinsics = self.cam_dict["extrinsics"]
        all_ids = self.cam_dict["ids"].tolist()
        all_centers = -all_extrinsics[:, :3, :3].permute(0, 2, 1) @ all_extrinsics[:, :3, -1:]

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
        plt.show()
        plt.close()

    def check_depth_existence(self):
        missing_depths = []
        for meta in tqdm.tqdm(self.metas, desc="Checking Depth Images"):
            scan_idx = meta["scan_idx"]
            cam_idx = meta["cam_idx"]
            ref_cam_idcs = meta["ref_cam_idcs"]
            scan_name = self.scan_list[scan_idx]

            all_cam_idcs = [cam_idx] + ref_cam_idcs
            all_cam_ids = [self.cam_dict["ids"][i] for i in all_cam_idcs]

            all_depth_paths = [self.data_dir / "Depths" / scan_name / self.get_depth_fname(i)
                               for i in all_cam_ids[1:]]

            for depth_path in all_depth_paths:
                if not depth_path.exists():
                    missing_depths.append(depth_path)
        if missing_depths:
            raise FileNotFoundError("Missing depth files", missing_depths)
