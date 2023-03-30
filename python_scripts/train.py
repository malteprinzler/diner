"""
Main script for training diner
"""

from omegaconf import OmegaConf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from src.data.pl_datamodule import PlDataModule
from pytorch_lightning import Trainer
from src.models.diner import DINER
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
from src.util.general import copy_python_files
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

def main():
    config_path = sys.argv[1]
    conf = OmegaConf.load(config_path)
    os.makedirs(conf.logger.kwargs.save_dir, exist_ok=True)
    datamodule = PlDataModule(conf.data.train, conf.data.val)
    datamodule.setup()

    # initialize model
    diner = DINER(nerf_conf=conf.nerf, renderer_conf=conf.renderer, znear=datamodule.train_set.znear,
                          zfar=datamodule.train_set.zfar, **conf.optimizer.kwargs)

    # initialize logger
    logger = TensorBoardLogger(**conf.logger.kwargs, name=None)

    # save configuration
    os.makedirs(logger.log_dir, exist_ok=True)
    os.system(f"cp {config_path} {os.path.join(logger.log_dir, 'config.yaml')}")
    copy_python_files("src", os.path.join(logger.log_dir, "code", "src"))
    copy_python_files("python_scripts", os.path.join(logger.log_dir, "code", "python_scripts"))

    # setting up checkpoint saver
    checkpoint_callback = ModelCheckpoint(**conf.checkpointing.kwargs, dirpath=logger.log_dir)

    # Setting up progress bar
    progress_bar = TQDMProgressBar()

    # initialize trainer
    trainer = Trainer(logger=logger, **conf.trainer.kwargs, callbacks=[checkpoint_callback, progress_bar])

    trainer.fit(diner, datamodule=datamodule, ckpt_path=conf.trainer.get("ckpt_path", None))


if __name__ == "__main__":
    main()
