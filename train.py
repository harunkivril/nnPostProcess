import sys
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import PostProcessPLDataset, return_model
from pytorch_lightning.loggers import WandbLogger
from evaluate import evaluate_results

AVAIL_GPUS = min(1, torch.cuda.device_count())
CONFIG_PATH = "/home/harunkivril/Workspace/BogaziciMS/MsThesis/NeuralPostProcess/hyperparams.yml"

if __name__=='__main__':
    model_name = sys.argv[1]

    config = OmegaConf.load(CONFIG_PATH)
    config = OmegaConf.merge(config["default"], config[model_name])
    config["test_start_year"] = min(config.test_years)

    print(config)
    seed_everything(3136)

    model = return_model(model_name, config)

    dataset = PostProcessPLDataset(
        batchsize=config.batchsize,
        val_years=config.val_years,
        test_years=config.test_years,
        transform_name=config.transform_name,
        meta_prefix=config.meta_prefix,
        era5_daily_prefix=config.era5_daily_prefix,
        gefs_daily_prefix=config.gefs_daily_prefix,
        dataloader_workers=config.dataloader_workers,
    )

    print(model)
    wandb_logger = WandbLogger(project="msthesis", group=model_name)

    early_stop_callback = EarlyStopping(
        monitor="Val Loss",
        min_delta=config.early_stop_delta,
        patience=config.early_stop_patience,
        verbose=False,
        mode="min"
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=config.max_epochs,
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[early_stop_callback]
    )

    trainer.fit(model, dataset)
    trainer.test(model, dataset)
    pred_gefs_mse, pred_mse = evaluate_results(
        scaler_name=config.transform_name,
        meta_prefix=config.meta_prefix,
        era5_daily_prefix=config.era5_daily_prefix,
        gefs_daily_prefix=config.gefs_daily_prefix,
        test_prediction_prefix=config.test_prediction_prefix,
    )
    wandb_logger.log_metrics({f"pred_gfs_mse":pred_gefs_mse})
    wandb_logger.log_metrics({f"pred_mse":pred_mse})
