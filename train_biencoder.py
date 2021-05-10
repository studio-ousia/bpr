from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from bpr.biencoder import BiEncoder


if __name__ == "__main__":
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument("--comet_offline", action="store_true")
    parent_parser.add_argument("--comet_project_name", type=str, default="biencoder")
    parent_parser.add_argument("--comet_save_dir", type=str, default="comet_logs")
    parent_parser.add_argument("--comet_workspace", type=str, default="wikipedia-qa")
    parent_parser.add_argument("--output_dir", type=str, default=".")
    parent_parser.add_argument("--tensorboard_name", type=str, default="biencoder")
    parent_parser.add_argument("--seed", type=int, default=1)
    parent_parser.add_argument("--use_comet", action="store_true")

    parser = BiEncoder.add_model_specific_args(parent_parser, root_dir=".")
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    pl.seed_everything(hparams.seed)

    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_avg_rank", mode="min")

    logger = TensorBoardLogger(save_dir=hparams.output_dir, name=hparams.tensorboard_name)
    if hparams.use_comet:
        comet_logger = CometLogger(
            save_dir=hparams.comet_save_dir,
            workspace=hparams.comet_workspace,
            project_name=hparams.comet_project_name,
            offline=hparams.comet_offline,
            auto_output_logging=False,
        )
        logger = [logger, comet_logger]

    trainer = Trainer.from_argparse_args(
        hparams,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=False,
        reload_dataloaders_every_epoch=True,
    )

    if trainer.num_gpus > 1 and hparams.distributed_backend != "ddp":
        raise RuntimeError("ddp needs to be used as the distributed backend when training the model with multiple GPUs")

    model = BiEncoder(hparams)
    trainer.fit(model)
