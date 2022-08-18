# Evaluate predictions of each variable in their original scale with MSE
import os
import torch
from tqdm import tqdm
from utils import MinMaxScaler, StandardScaler

def evaluate_results(
    scaler_name,
    era5_daily_prefix,
    gefs_daily_prefix,
    test_prediction_prefix,
    meta_prefix,
    gefs_var_names = ('t', 'u', 'v', 'w'),
    era5_var_names = ('r', 't', 'u', 'v', 'w'),
    scaled_perf = False,
    eliminate_corners = False,
):
    if scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler(meta_prefix)
    elif scaler_name == "StandardScaler":
        scaler = StandardScaler(meta_prefix)
    else:
        raise ValueError("Scaler not valid. Options: MinMaxScaler, StandardScaler")

    gefs_error = 0
    prediction_full_error = 0
    prediction_gefs_dims_error = 0
    n_instance = 0
    for pred_date in tqdm(os.listdir(test_prediction_prefix)):
        n_instance += 1
        daily_gefs_path = f"{gefs_daily_prefix}/{pred_date}"
        daily_era5_path = f"{era5_daily_prefix}/{pred_date}"
        daily_pred_path = f"{test_prediction_prefix}/{pred_date}"

        era5 = torch.load(daily_era5_path)
        prediction = torch.load(daily_pred_path).cpu()
        gefs = torch.load(daily_gefs_path)

        if scaled_perf:
            era5 = scaler.transform(era5, "era5")
            gefs = scaler.transform(gefs, "gefs")
        else:
            prediction = scaler.inverse_transform(prediction, "era5")

        if eliminate_corners:
            era5 = era5[:, :, :, :, 1:-1, 1:-1 ]
            prediction = prediction[:, :, :, :, 1:-1, 1:-1 ]
            gefs = gefs[:, :, :, 1:-1, 1:-1]

        prediction_full_error += torch.square(era5 - prediction).mean(dim=(0,1,3,4,5))
        # Reduce to gefs dimensions, remove relative humidity
        # Original: [datetime, 3 hour roll, variable, pressure_level, lat, lon]
        era5 = era5[:, 0, 1:, [0,1,2,3,4,6,8], :, :].squeeze()
        prediction = prediction[:, 0, 1:, [0,1,2,3,4,6,8], :, :].squeeze()
        prediction_gefs_dims_error += torch.square(era5 - prediction).mean(dim=(0,2,3,4))

        # Remove specific humidiy
        # [datetime, variable, pressure_level, lat, lon]
        gefs = gefs[:, 1:, :, :, :]
        gefs_error += torch.square(era5 - gefs).mean(dim=(0,2,3,4))

    gfs_mse = (gefs_error/n_instance).numpy()
    gfs_mse = dict(zip(gefs_var_names, list(gfs_mse)))
    pred_gefs_mse = (prediction_gefs_dims_error/n_instance).numpy()
    pred_gefs_mse = dict(zip(gefs_var_names, list(pred_gefs_mse)))
    pred_mse = (prediction_full_error/n_instance).numpy()
    pred_mse = dict(zip(era5_var_names, list(pred_mse)))

    print(f"GEFS MSE: {gfs_mse}")
    print(f"PRED GEFS DIM MSE: {pred_gefs_mse}")
    print(f"PRED FULL MSE: {pred_mse}")

    return pred_gefs_mse, pred_mse
