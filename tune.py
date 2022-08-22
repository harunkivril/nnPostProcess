import optuna
import torch
import wandb
import joblib
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from optuna.samplers import RandomSampler, TPESampler

from utils import PostProcessPLDataset, create_search_space, return_model, return_initial_params

MODEL_NAME = "fully_conv_model"
SAVE_NAME = "fully_conv_model4"
MAX_TIME = 2*24*60*60 # 2 Day
#SAMPLER = TPESampler(seed=3136)
SAMPLER = RandomSampler(seed=3136)

AVAIL_GPUS = min(1, torch.cuda.device_count())
CONFIG_PATH = "/home/harunkivril/Workspace/BogaziciMS/MsThesis/NeuralPostProcess/hyperparams.yml"
SAVE_DIR = "/home/harunkivril/Workspace/BogaziciMS/MsThesis/wandb"

config = OmegaConf.load(CONFIG_PATH)
config = OmegaConf.merge(config["default"], config[MODEL_NAME])
config["test_start_year"] = min(config.test_years)

initial_params = return_initial_params(MODEL_NAME)

def objective(trial):

    params = create_search_space(MODEL_NAME, trial)
    params = OmegaConf.merge(config, params)
    seed_everything(3136)

    model = return_model(MODEL_NAME, params)
    wandb_logger = WandbLogger(
        project="msthesis_tune",
        group=SAVE_NAME,
        save_dir=SAVE_DIR
    )

    dataset = PostProcessPLDataset(
        batchsize=params.batchsize,
        val_years=params.val_years,
        test_years=params.test_years,
        transform_name=params.transform_name,
        meta_prefix=params.meta_prefix,
        era5_daily_prefix=params.era5_daily_prefix,
        gefs_daily_prefix=params.gefs_daily_prefix,
        dataloader_workers=params.dataloader_workers,
    )

    early_stop_callback = EarlyStopping(
        monitor="Val Loss",
        min_delta=params.early_stop_delta,
        patience=params.early_stop_patience,
        verbose=False,
        mode="min"
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=params.max_epochs,
        log_every_n_steps=params.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[early_stop_callback],
        enable_checkpointing=False,
    )

    try:
        trainer.fit(model, dataset)
        val_loss = trainer.logged_metrics["Val Loss"].cpu().numpy()
    except RuntimeError:
        print("WARNING GPU MEM IS NOT ENOUGH FOR PARAM SETTING")
        val_loss = 1

    wandb.finish()

    return val_loss

storage_name = f"sqlite:///{SAVE_NAME}.db"
study = optuna.create_study(
    study_name=SAVE_NAME,
    storage=storage_name,
    load_if_exists=True,
    direction="minimize",
    sampler=SAMPLER,
)
if initial_params:
    study.enqueue_trial(initial_params)
study.optimize(objective, timeout=MAX_TIME)

joblib.dump(study, f"{config.csv_save_path}/{SAVE_NAME}_study.pkl")

print(f"Number of finished trials: {len(study.trials)}")

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
