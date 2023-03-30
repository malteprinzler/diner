import tqdm
from pytorch_lightning import LightningModule
from src.util.import_helper import import_obj
from src.util.cam_geometry import gen_rays
from src.util.general import *
from pytorch_lightning.utilities import move_data_to_device
from src.util import torch_helpers
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os
from src.evaluation import eval_suite
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from random import Random
from src.losses import VGGLoss, AntibiasLoss
import torch


class DINER(LightningModule):
    def __init__(self, nerf_conf, renderer_conf, znear, zfar, ray_batch_size=128, lr=1e-4, img_log_interval=10000,
                 n_samples_score_eval=100, cam_sweep_settings=dict(), w_vgg=0., vgg_spatch=64,
                 w_antibias=0., antibias_downsampling=3):
        """
        DINER Optimizer

        Parameters
        ----------
        nerf_conf: Omegaconf configs for nerf (see example config files)
        renderer_conf: Omegaconf configs for renderer (see example config files)
        znear
        zfar
        ray_batch_size: how many rays to evaluate per sample in batch, set to patch_size^2 if w_perc != 0
        lr
        img_log_interval: image log interval (in steps)
        n_samples_score_eval: how many images to evaluate the validation scores on
        cam_sweep_settings: dict with camera sweep settings (see example config files)
        w_vgg: weight for perceptual loss
        vgg_spatch: patch size for perceptual loss
        w_antibias: weight for antibias loss
        antibias_downsampling: downsampling exponent for antibias loss. 3 corresponds to 2^3 = 8 fold downsampling
        """

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.nerf = import_obj(nerf_conf.module)(**nerf_conf.kwargs)
        self.renderer = import_obj(renderer_conf.module)(**renderer_conf.kwargs)
        self.cam_sweep_settings = cam_sweep_settings

        self.img_log_interval = img_log_interval
        self.n_samples_score_eval = n_samples_score_eval
        self.lr = lr
        self.w_vgg = w_vgg
        self.vgg_spatch = vgg_spatch
        self.w_antibias = w_antibias
        self.ray_batch_size = ray_batch_size if self.w_vgg == 0 else vgg_spatch ** 2
        self.register_buffer("znear", torch.tensor(znear))
        self.register_buffer("zfar", torch.tensor(zfar))

        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.vggloss = VGGLoss() if self.w_vgg != 0 else None
        self.antibiasloss = AntibiasLoss(n_downsampling=antibias_downsampling) if self.w_antibias != 0 else None

    def encode_batch(self, batch):
        self.nerf.encode(images=batch["src_rgbs"],
                         depths=batch["src_depths"],
                         depths_std=batch["src_depth_stds"],
                         extrinsics=batch["src_extrinsics"],
                         intrinsics=batch["src_intrinsics"])

    def predict_imgs_from_batch(self, batch, return_depth=False):
        SB, _, H, W = batch["target_rgb"].shape
        self.encode_batch(batch)

        # generate rays
        znear = self.znear.expand(SB)
        zfar = self.zfar.expand(SB)
        rays = gen_rays(extrinsics=batch["target_extrinsics"], intrinsics=batch["target_intrinsics"], W=W, H=H,
                        z_near=znear, z_far=zfar)  # (SB, H, W, 8)
        rays = rays.view(SB, H * W, -1)

        # rendering
        rgb, depth = list(), list()
        for ray_batch in torch.split(rays, self.ray_batch_size, dim=1):
            render_dict = self.renderer.forward(model=self.nerf, rays=ray_batch.contiguous())
            rgb.append(render_dict.fine.rgb)
            depth.append(render_dict.fine.depth)
        rgb = torch.cat(rgb, dim=1)
        depth = torch.cat(depth, dim=1)
        rgb = rgb.view(SB, H, W, 3).permute(0, 3, 1, 2)
        depth = depth.view(SB, H, W, 1).permute(0, 3, 1, 2)

        if return_depth:
            return rgb, depth
        else:
            return rgb

    @torch.no_grad()
    def create_prediction_folder(self, outdir, return_last_pred=False, dataloader=None, show_tqdm=False):

        os.makedirs(outdir, exist_ok=True)

        # creating dataloader
        if dataloader is None:
            dataset = self.trainer.datamodule.val_set
            datalen = len(dataset)
            example_loader = self.trainer.datamodule.val_dataloader()
            batch_size = example_loader.batch_size
            num_workers = 0
            sample_idcs = list(range(datalen))
            if self.n_samples_score_eval > 0 and self.n_samples_score_eval < datalen:
                sample_idcs = Random(0).sample(sample_idcs, self.n_samples_score_eval)
            sample_idcs = torch.tensor(sample_idcs).int()
            batch_sampler = BatchSampler(sample_idcs, batch_size=batch_size, drop_last=False)
            dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)

        # predicting and writing images & data
        iterator = tqdm.tqdm(dataloader, total=len(dataloader), mininterval=30.) if show_tqdm else dataloader
        for batch in iterator:
            batch = move_data_to_device(batch, self.device)
            imgs, depths = self.predict_imgs_from_batch(batch, return_depth=True)
            depths = torch_helpers.torch_cmap(depths)
            src_imgs = torch.cat(batch["src_rgbs"].unbind(1), dim=-1)  # (B, 3, H, W')
            gt_imgs = batch["target_rgb"]
            stems = batch["sample_name"]

            for i in range(len(imgs)):
                save_image(imgs[i], os.path.join(outdir, stems[i] + eval_suite.PRED_SUFFIX))
                save_image(depths[i],
                           os.path.join(outdir, stems[i] + eval_suite.DEPTH_SUFFIX))
                save_image(src_imgs[i], os.path.join(outdir, stems[i] + eval_suite.REF_SUFFIX))
                save_image(gt_imgs[i], os.path.join(outdir, stems[i] + eval_suite.GT_SUFFIX))

        if return_last_pred:
            return dict(pred_rgb=imgs, pred_depth=depths, gt_rgb=gt_imgs, src_rgbs=src_imgs)

    @torch.no_grad()
    def create_cam_sweep(self, outdir, nframes=30, n_cam_sweeps=4, fps=5, sample_idcs=None, dataset=None):
        import tqdm
        os.makedirs(outdir, exist_ok=True)

        dataset = self.trainer.datamodule.val_set if dataset is None else dataset
        cam_sweep_idcs = torch.linspace(0, len(dataset) - 1, n_cam_sweeps).int() if sample_idcs is None else sample_idcs

        for idx in cam_sweep_idcs:
            base_sample = move_data_to_device(dataset[idx], self.device)
            H, W = base_sample["target_rgb"].shape[-2:]

            base_batch = torch_helpers.unsqueeze_dict(base_sample)
            self.encode_batch(base_batch)
            target_extrinsics = dataset.get_cam_sweep_extrinsics(nframes=nframes,  # (N, 4, 4)
                                                                 scan_idx=idx)
            target_extrinsics = target_extrinsics.to(self.device)

            # # visualize camera sweep extrinsics
            # all_extrinsics = torch.cat((target_extrinsics, base_batch["src_extrinsics"][0]), dim=0)
            # all_poses = torch.linalg.inv(all_extrinsics).cpu()
            # fig = plt.figure()
            # ax = fig.add_subplot(projection="3d")
            # s = .1
            # for i, color in enumerate(["red", "green", "blue"]):
            #     ax.quiver(all_poses[:, 0, -1], all_poses[:, 1, -1], all_poses[:, 2, -1],
            #               s * all_poses[:, 0, i], s * all_poses[:, 1, i], s * all_poses[:, 2, i],
            #               edgecolor=color)
            # ax.set_xlabel("X")
            # ax.set_ylabel("Y")
            # ax.set_zlabel("Z")
            # ax.set_xlim(-1.5, 1.5)
            # ax.set_ylim(-1.5, 1.5)
            # ax.set_zlim(-1.5, 1.5)
            # plt.show()
            # plt.close()

            rgbs = list()
            depths = list()

            for i in tqdm.tqdm(range(nframes)):
                # generate rays
                rays = gen_rays(extrinsics=target_extrinsics[i:i + 1],
                                intrinsics=base_batch["target_intrinsics"],
                                W=W, H=H,
                                z_near=self.znear, z_far=self.zfar)  # (cam_sweep_frames, H, W, 8)
                rays = rays.view(1, H * W, -1).clone()
                rays = torch.split(rays, self.ray_batch_size, dim=1)

                # rendering
                rgbs_ = list()
                depths_ = list()
                for rays_batch in rays:
                    render_dict = self.renderer.forward(model=self.nerf, rays=rays_batch)
                    rgb = render_dict.fine.rgb
                    depth = render_dict.fine.depth
                    rgb = rgb.view(-1, 3).cpu()
                    depth = depth.view(-1, 1).cpu()
                    rgbs_.append(rgb)
                    depths_.append(depth)
                rgbs.append(torch.cat(rgbs_, dim=0).view(H, W, 3).permute(2, 0, 1))
                depths.append(torch_helpers.torch_cmap(torch.cat(depths_, dim=0).view(H, W, 1).permute(2, 0, 1)))
            rgbs = torch.stack(rgbs, dim=0)
            depths = torch.stack(depths, dim=0)

            # saving
            vid_name = f"{base_sample['sample_name']}.mp4"
            srcimg_name = vid_name.replace(".mp4", "-ref_imgs.jpg")
            vidpath = os.path.join(outdir, vid_name)
            srcimg_path = os.path.join(outdir, srcimg_name)

            frames = torch.cat((rgbs, depths), dim=-2)
            idcs = torch.cat((torch.arange(nframes), torch.arange(nframes - 1, 0, -1)))
            frames = frames[idcs]
            src_imgs = torch.cat(base_sample['src_rgbs'].unbind(dim=0), dim=-1)

            torch_helpers.save_torch_video(frames, vidpath, fps)
            save_image(src_imgs, srcimg_path)

    def calc_losses(self, batch):
        SB, _, H, W = batch["target_rgb"].shape
        self.encode_batch(batch)

        # generate rays
        znear = self.znear.expand(SB)
        zfar = self.zfar.expand(SB)
        rays = gen_rays(extrinsics=batch["target_extrinsics"],
                        intrinsics=batch["target_intrinsics"],
                        W=W, H=H,
                        z_near=znear, z_far=zfar)  # (SB, H, W, 8)

        if self.w_vgg == 0.:
            pix_idcs = torch.randint(0, H * W, (SB, self.ray_batch_size))
        else:
            # sampling patch
            fg_mask = batch["target_alpha"][:, 0].clone()  # N, H, W
            pad = (self.vgg_spatch + 1) // 2
            fg_mask[..., :pad] = 0
            fg_mask[..., :pad, :] = 0
            fg_mask[..., -pad:] = 0
            fg_mask[..., -pad:, :] = 0
            patch_centers = torch.multinomial(fg_mask.view(SB, H * W), 1)  # N
            patch_centers = torch.cat((patch_centers % W, torch.div(patch_centers, W, rounding_mode="floor")), dim=-1)  # N, 2 (x,y)
            pix_coords = torch.stack(
                torch.meshgrid(torch.arange(self.vgg_spatch, device=self.device),  # h, w, 2 (x, y)
                               torch.arange(self.vgg_spatch, device=self.device))[::-1], dim=-1)
            pix_coords -= pad
            pix_coords = patch_centers.unsqueeze(1).unsqueeze(1) + pix_coords.unsqueeze(0)  # N, h, w, 2 (x, y)
            pix_idcs = pix_coords[..., 0] + pix_coords[..., 1] * W  # N, h, w
            pix_idcs = pix_idcs.flatten(start_dim=1)  # N, B

            # # visualizing sampled patch
            # import matplotlib.pyplot as plt
            # fg_mask_ = fg_mask[0].flatten()
            # fg_mask_[pix_idcs[0]] = 2
            # fg_mask_ = fg_mask_.view(H, W)
            # plt.imshow(fg_mask_.cpu())
            # plt.show()

        batch_idx_helper = torch.arange(SB).unsqueeze(-1).expand(-1, self.ray_batch_size)
        rays = rays.view(SB, H * W, -1)[batch_idx_helper, pix_idcs]  # (SB, B, 8)

        # predict colors
        render_dict = self.renderer.forward(model=self.nerf, rays=rays)
        pred_fine = render_dict.fine.rgb

        # calculate raycolors
        gt_colors = batch["target_rgb"].view(SB, 3, -1).permute(0, 2, 1)[batch_idx_helper, pix_idcs]  # (SB, B, 3)

        loss_fine = self.criterion(pred_fine, gt_colors)
        loss_total = loss_fine

        # perceptual loss
        if self.w_vgg > 0:
            loss_vgg = self.vggloss(pred_fine.view(SB, self.vgg_spatch, self.vgg_spatch, 3).permute(0, 3, 1, 2),
                                    gt_colors.view(SB, self.vgg_spatch, self.vgg_spatch, 3).permute(0, 3, 1, 2))
            loss_total += self.w_vgg * loss_vgg
        else:
            loss_vgg = 0.

        # antibias loss
        if self.w_antibias > 0:
            loss_antibias = self.antibiasloss(
                pred_fine.view(SB, self.vgg_spatch, self.vgg_spatch, 3).permute(0, 3, 1, 2),
                gt_colors.view(SB, self.vgg_spatch, self.vgg_spatch, 3).permute(0, 3, 1, 2))
            loss_total += self.w_antibias * loss_antibias
        else:
            loss_antibias = 0.

        loss_dict = dict(rgb_fine=loss_fine, vgg_fine=loss_vgg, antibias=loss_antibias,
                         total=loss_total)

        return loss_dict

    def training_step(self, batch, batch_idx):
        # loss calculation
        loss_dict = self.calc_losses(batch)
        loss_dict["step"] = float(self.global_step)
        batch_size = batch["target_rgb"].shape[0]
        self.log_dict(loss_dict, on_step=True, batch_size=batch_size)

        return loss_dict["total"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_dict = self.calc_losses(batch)
        log_dict = prefix_dict_keys(loss_dict, "val_")
        log_dict["step"] = float(self.global_step)
        batch_size = batch["target_rgb"].shape[0]
        self.log_dict(log_dict, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss_dict["total"]

    @rank_zero_only
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        if self.global_step > 0:
            eval_dir = os.path.join(self.trainer.log_dir, f"eval_{self.trainer.global_step:06d}")
            os.makedirs(eval_dir, exist_ok=True)

            # saving checkpoint
            self.trainer.save_checkpoint(os.path.join(eval_dir, f"{self.trainer.global_step:06d}.ckpt"))

            # creating and evaluating visualizations
            visdir = os.path.join(eval_dir, "visualizations")
            self.create_prediction_folder(outdir=visdir)
            score_dict = eval_suite.evaluate_folder(visdir, eval_dir)
            log_dict = prefix_dict_keys(score_dict, "valscores_")
            log_dict["step"] = float(self.global_step)
            self.log_dict(log_dict, rank_zero_only=True, sync_dist=True)

            # creating cam_sweeps
            cam_sweep_dir = os.path.join(eval_dir, "cam_sweeps")
            self.create_cam_sweep(cam_sweep_dir, **self.cam_sweep_settings)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nerf.parameters(), lr=self.lr)
        return optimizer
