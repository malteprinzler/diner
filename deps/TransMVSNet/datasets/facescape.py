from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from pathlib import Path
import json
import matplotlib.pyplot as plt
import tqdm


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    RGBA_FNAME = "rgba_colorcalib.png"
    DEPTH_FNAME = "depth.png"
    znear = 1.
    zfar = 2.5

    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = Path(datapath)
        self.mode = mode
        self.stage = self.mode
        self.nviews = nviews
        assert nviews == 4
        self.ndepths = ndepths
        self.kwargs = kwargs
        self.range_hor = 45
        self.range_vert = 30
        self.slide_range = 40
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test", "write_prediction"]
        self.metas = self.build_list()

    def build_list(self):

        meta_dir = Path("assets/data_splits/facescape")

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
            meta_fpath = meta_dir / (stage + "_" + str(self.range_hor) + "_" + str(self.range_vert) +
                                     (f"_{str(self.slide_range)}" if self.slide_range != 0 else "") + ".txt")
            with open(meta_fpath, "r") as f:
                diner_metas_ = json.load(f)
            diner_metas += diner_metas_

        # converting diner metas to MaGNet-usable metas
        metas = list()
        old_scan_path = ""
        old_ref_ids = ""
        sample_idx = 0
        for meta in diner_metas:
            if meta["scan_path"] != old_scan_path or str(meta["ref_ids"]) != old_ref_ids:
                old_scan_path = meta["scan_path"]
                old_ref_ids = str(meta["ref_ids"])
                assert self.nviews == len(meta["ref_ids"])
                for i in range(self.nviews):
                    ref_ids = [r[:1] for r in meta["ref_ids"][:i]] + \
                              [r[:1] for r in meta["ref_ids"][i + 1:]]
                    sample_meta = dict(idx=sample_idx,
                                       scan_path=meta["scan_path"],
                                       target_ids=meta["ref_ids"][i][:1],  # (noptions,)
                                       ref_ids=ref_ids)  # (n_views-1, noptions)
                    metas.append(sample_meta)
                    sample_idx += 1
        return metas

    def __len__(self):
        return len(self.metas)

    def read_img(self, img_path):
        """
        returns H, W, 3 and H, W, 1
        """
        img = Image.open(img_path).convert("RGBA")
        img = np.array(img).astype(np.float32) / 255.0  # (H, W, 4)
        mask = img[..., -1:] > .5
        img[~mask[..., 0], :3] = 1.  # white bg
        np_img = img[..., :3]
        mask = mask.astype(np.float32)
        return np_img, mask

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
        frame, subject = scan_path.name, scan_path.parent.name
        return frame, subject

    @staticmethod
    def int_2_viewname(i: int):
        return f"view_{i:05d}"

    def __getitem__(self, idx):
        import json
        meta = self.metas[idx]

        target_id = np.random.choice(meta["target_ids"])
        ref_ids = np.array([np.random.choice(meta["ref_ids"][i]) for i in range(self.nviews - 1)])

        scan_path = self.datapath / meta["scan_path"]

        # use only the reference view and first nviews-1 source views
        view_ids = [target_id] + ref_ids.tolist()

        cam_path = scan_path / "cameras.json"
        with open(cam_path, "r") as f:
            cam_dict = json.load(f)

        imgs = []
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            view_path = scan_path / self.int_2_viewname(int(vid))
            img_path = view_path / "rgba_colorcalib.png"

            img, mask = self.read_img(img_path)

            extrinsics = cam_dict[vid]["extrinsics"] + [[0., 0., 0., 1.]]
            extrinsics = np.array(extrinsics, dtype=np.float32)
            intrinsics = cam_dict[vid]["intrinsics"]
            intrinsics = np.array(intrinsics, dtype=np.float32)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                dmap_path = view_path / "depth.png"
                depth = self.read_depth(dmap_path)
                mask_ms = self.multiscale_x(mask[..., 0])
                depth_ms = self.multiscale_x(depth[..., 0])

                # get depth values
                depth_values = np.linspace(self.znear, self.zfar, self.ndepths, dtype=np.float32)

            imgs.append(img)

        # all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2

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


if __name__ == "__main__":
    ds = MVSDataset("/is/rg/ncs/datasets/FACESCAPE/FACESCAPE_PROCESSED", None, mode="train", nviews=4, )
    # ds.find_depth_range()

    for i in np.random.permutation(np.arange(len(ds))):
        ds.visualize_item(i)

    ds.visualize_item(0)
    # ds.visualize_item(0)
    # print(len(ds))
    #
    # import tqdm
    #
    # for i in tqdm.tqdm(np.random.permutation(np.arange(len(ds)))):
    #     meta = ds.metas[i]
    #     scan, light_idx, ref_view, src_views = meta
    #     img_path = os.path.join(ds.datapath,
    #                             'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, ref_view + 1, light_idx))
    #     assert os.path.exists(img_path)
