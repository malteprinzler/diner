from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
import matplotlib.pyplot as plt
import torch


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        assert nviews == 4
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.kwargs = kwargs
        print("mvsdataset kwargs", self.kwargs)

        assert self.mode in ["train", "val", "test", "write_prediction"]
        self.metas = self.build_list()

    def get_target_and_ref_ids(self):
        # Only works for cam grid as in DTU!

        tl = torch.tensor([10, 0, 1, 2, 9, 13, 12, 11, 10])
        bl = torch.tensor([30, 27, 26, 25, 31, 45, 46, 47, 29])
        tr = torch.tensor([6, 2, 3, 4, 5, 18, 17, 16, 7])
        br = torch.tensor([35, 22, 21, 20, 36, 40, 41, 42, 34])

        if self.mode != "train":  # for validation, only use relevant reference views
            tl = tl[:1]
            bl = bl[:1]
            tr = tr[:1]
            br = br[:1]

        targets = torch.cat((tl, bl, tr, br))
        srcs = torch.cat((torch.stack((bl, tr, br), dim=-1),
                          torch.stack((tl, tr, br), dim=-1),
                          torch.stack((tl, bl, br), dim=-1),
                          torch.stack((tl, bl, tr), dim=-1)), dim=0)
        return targets, srcs

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        # for scan in scans:
        #     pair_file = "Cameras/pair.txt"
        #     # read the pair file
        #     with open(os.path.join(self.datapath, pair_file)) as f:
        #         num_viewpoint = int(f.readline())
        #         # viewpoints (49)
        #         for view_idx in range(num_viewpoint):
        #             ref_view = int(f.readline().rstrip())
        #             src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
        #             # light conditions 0-6
        #             for light_idx in range(7):
        #                 breakpoint()
        #                 metas.append((scan, light_idx, ref_view, src_views))

        targets, srcs = self.get_target_and_ref_ids()
        # scans
        for scan in scans:
            for target_id, src_ids in zip(targets, srcs):
                for light_idx in range(7) if self.mode == "train" else [3]:  # only uniform lighting for validation
                    metas.append((scan, light_idx, target_id.item(), src_ids.tolist()))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def prepare_img(self, hr_img):
        # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

        # downsample
        h, w = hr_img.shape
        hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        # crop
        h, w = hr_img_ds.shape
        target_h, target_w = 512, 640
        start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
        hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

        return hr_img_crop

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": np_img,
        }
        return np_img_ms

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_depth_hr(self, filename):
        # read pfm depth file
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            "stage1": cv2.resize(depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            img = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask_hr(mask_filename_hr)
                depth_ms = self.read_depth_hr(depth_filename_hr)

                # get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

            imgs.append(img)

        # all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "dpath": 'Depths/{}/depth_map_{:0>4}.pfm'.format(scan, ref_view),
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "depth_interval": depth_interval,
                "mask": mask}

    def visualize_cam_grid(self, show=True):
        cam_paths = [os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt'.format(vid)) for vid in range(49)]
        extrinsics = torch.from_numpy(np.stack([self.read_cam_file(f)[1] for f in cam_paths]))

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
        s = 50.
        for i, color in enumerate(["red", "green", "blue"]):
            ax.quiver(all_centers[:, 0, 0], all_centers[:, 1, 0], all_centers[:, 2, 0],
                      s * all_extrinsics[:, i, 0], s * all_extrinsics[:, i, 1], s * all_extrinsics[:, i, 2],
                      edgecolor=color)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.set_xlim((-1.5, 1.5))
        # ax.set_ylim((-1.5, 1.5))
        # ax.set_zlim((-1.5, 1.5))

        # checking reprojections
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3))
        random_points = np.random.normal(size=(10, 3)) * 100 + np.array([[0., 0., 300]])
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


if __name__ == "__main__":
    ds = MVSDataset("data/DTU", "lists/dtu/train.txt", mode="train", nviews=4, )
    ds.visualize_item(0)
    print(len(ds))

    import tqdm

    for i in tqdm.tqdm(np.random.permutation(np.arange(len(ds)))):
        meta = ds.metas[i]
        scan, light_idx, ref_view, src_views = meta
        img_path = os.path.join(ds.datapath,
                                'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, ref_view + 1, light_idx))
        assert os.path.exists(img_path)
