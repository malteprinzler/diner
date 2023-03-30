import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from omegaconf import OmegaConf
import torch
from torch.utils.data.dataloader import DataLoader
from src.models.diner import DINER
from src.evaluation.eval_suite import evaluate_folder
from pathlib import Path
from argparse import ArgumentParser
from random import Random
from src.util.import_helper import import_obj

parser = ArgumentParser(description="Loads trained DINER checkpoint and creates folder with predictions from the validation "
                                    "set and performs quantitative evaluation on predictions.")
parser.add_argument("--config", type=Path, default=Path("configs/evaluate_diner_on_facescape.yaml"))
parser.add_argument("--ckpt", type=Path, default=Path("assets/ckpts/facescape/DINER.ckpt"))
parser.add_argument("--out", type=Path, default=Path("outputs/facescape/diner_full_evaluation"))
parser.add_argument("--nsamples", type=int, default=-1, help="samples per ray, -1 (default) uses same as in checkpoint")
parser.add_argument("--n", type=int, default=-1, help="number of dataset samples to evaluate on, -1 (default) evaluates all")
args = parser.parse_args()

config_path = args.config
ckpt_path = args.ckpt
out_path = args.out
vis_path = out_path / "visualizations"
n = args.n

conf = OmegaConf.load(config_path)
dset_class = import_obj(conf.data.val.dataset.module)
dataset = dset_class(**conf.data.val.dataset.kwargs, stage="val")
datalen = len(dataset)
sample_idcs = list(range(datalen))
if n > 0 and n < datalen:
    sample_idcs = Random(0).sample(sample_idcs, n)
sample_idcs = torch.tensor(sample_idcs).int()
dataloader = DataLoader(dataset, sampler=sample_idcs, drop_last=False, **conf.data.val.dataloader.kwargs)
ckpt = torch.load(ckpt_path)
state_dict = ckpt["state_dict"]
diner = DINER.load_from_checkpoint(ckpt_path).cuda().eval()
diner.znear = torch.tensor(dataset.znear, device=diner.device)
diner.zfar = torch.tensor(dataset.zfar, device=diner.device)
upsample_rate = args.nsamples / diner.renderer.n_samples
if args.nsamples > 0:
    diner.renderer.n_samples = args.nsamples
    diner.renderer.n_gaussian = int(diner.renderer.n_gaussian * upsample_rate)
diner.create_prediction_folder(vis_path, dataloader=dataloader, show_tqdm=True)
evaluate_folder(vis_path, out_path)
