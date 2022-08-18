import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import MinMaxScaler, StandardScaler
from omegaconf import OmegaConf

extract_era5 = True
extract_gefs = True
scaler_name = "MinMaxScaler"

CONFIG_PATH = "/home/harunkivril/Workspace/BogaziciMS/MsThesis/NeuralPostProcess/hyperparams.yml"
config = OmegaConf.load(CONFIG_PATH)["default"]
config.meta_prefix = "/media/harunkivril/HDD/MsThesis/FinalResults/meta"
config.test_prediction_prefix = "/media/harunkivril/HDD/MsThesis/FinalResults/fc/test_predictions"
eligible_farm_path = "/media/harunkivril/HDD/MsThesis/eligible_farms.csv"

era5_pres_idx = [0, 1, 2]
gefs_pres_idx = [0, 1, 2]

def shift_time(data_df, n_hours):
    data_df["time"] = pd.to_datetime(data_df.date) + pd.to_timedelta(data_df.hour + n_hours, "h")
    data_df["date"] = data_df.time.dt.date
    data_df["hour"] = data_df.time.dt.hour
    data_df = data_df.drop(columns="time")
    return data_df

def extract_ws_features(data_df, ugrd_cols):
    vgrd_cols = [x.replace("UGRD", "VGRD") for x in ugrd_cols]
    ws_cols = [x.replace("UGRD", "WS") for x in ugrd_cols]
    ws2_cols = [x.replace("UGRD", "WS2") for x in ugrd_cols]
    ws3_cols = [x.replace("UGRD", "WS3") for x in ugrd_cols]
    wdir_cols = [x.replace("UGRD", "WDIR") for x in ugrd_cols]
    data_df[ws_cols] = (data_df[ugrd_cols].values**2 + data_df[vgrd_cols].values**2)**0.5
    data_df[ws2_cols] = data_df[ws_cols].values**2
    data_df[ws3_cols] = data_df[ws_cols].values**3
    data_df[wdir_cols] = np.arctan2(data_df[ugrd_cols].values, data_df[vgrd_cols].values)
    return data_df

def get_era5(bbox, pres_idx):
    # bbox has to be 2x2  square
    left, right, down, up = bbox["left"], bbox["right"], bbox["down"], bbox["up"]

    vgrd_cols = [f"ERA5_VGRD_PRS_LOC{i}" for i in range(4)]
    ugrd_cols = [f"ERA5_UGRD_PRS_LOC{i}" for i in range(4)]
    pres_idx = [pres_idx] if isinstance(pres_idx, int) else pres_idx

    era5_df = []
    for pred_date in tqdm(os.listdir(config.test_prediction_prefix)):

        current_date = pred_date.split(".")[0]
        next_day = pd.to_datetime(current_date) + pd.to_timedelta(1, "d")
        next_day = next_day.strftime("%Y-%m-%d")

        dates = [current_date]*21 + [next_day]*3
        hours = list(range(3, 24)) + list(range(3))

        daily_era5_path = f"{config.era5_daily_prefix}/{pred_date}"
        era5 = torch.load(daily_era5_path)

        all_pressures = None
        for idx in pres_idx:

            era5_pres_u = era5[:, :, 2, idx, up:down+1, left:right+1].squeeze()
            era5_pres_v = era5[:, :, 3, idx, up:down+1, left:right+1].squeeze()

            era5_pres_u = era5_pres_u.view(24, 2, 2).numpy().reshape((24, 4))
            era5_pres_v = era5_pres_v.view(24, 2, 2).numpy().reshape((24, 4))

            pres_ugrd_cols = [x.replace("_PRS_", f"_PRS{idx}_") for x in ugrd_cols]
            pres_vgrd_cols = [x.replace("_PRS_", f"_PRS{idx}_") for x in vgrd_cols]

            era5_pres_df = pd.DataFrame(era5_pres_u, columns=pres_ugrd_cols)
            era5_pres_df["date"] = dates
            era5_pres_df["hour"] = hours
            era5_pres_df[pres_vgrd_cols] = era5_pres_v

            if all_pressures is None:
                all_pressures = era5_pres_df
            else:
                all_pressures = all_pressures.merge(era5_pres_df)

        era5_df.append(all_pressures)

    era5_df = pd.concat(era5_df, ignore_index=True)

    ugrd_cols = [x for x in era5_df if "UGRD" in x]
    era5_df = extract_ws_features(era5_df, ugrd_cols)
    return era5_df

def get_gefs(bbox, pres_idx):
    left, right, down, up = bbox["left"], bbox["right"], bbox["down"], bbox["up"]
    gefs_df = []
    vgrd_cols = [f"GEFS_VGRD_PRS_LOC{i}" for i in range(4)]
    ugrd_cols = [f"GEFS_UGRD_PRS_LOC{i}" for i in range(4)]
    pres_idx = [pres_idx] if isinstance(pres_idx, int) else pres_idx

    for pred_date in tqdm(os.listdir(config.test_prediction_prefix)):

        current_date = pred_date.split(".")[0]
        next_day = pd.to_datetime(current_date) + pd.to_timedelta(1, "d")
        next_day = next_day.strftime("%Y-%m-%d")

        dates = [current_date]*7 + [next_day]
        hours = list(range(3, 24, 3)) + [0]

        daily_gefs_path = f"{config.gefs_daily_prefix}/{pred_date}"
        gefs = torch.load(daily_gefs_path)

        all_pressures = None
        for idx in pres_idx:

            gefs_pres_u = gefs[:, 2, idx, up:down+1, left:right+1].squeeze()
            gefs_pres_v = gefs[:, 3, idx, up:down+1, left:right+1].squeeze()

            gefs_pres_u = gefs_pres_u.view(8, 2, 2).numpy().reshape((8, 4))
            gefs_pres_v = gefs_pres_v.view(8, 2, 2).numpy().reshape((8, 4))

            pres_ugrd_cols = [x.replace("_PRS_", f"_PRS{idx}_") for x in ugrd_cols]
            pres_vgrd_cols = [x.replace("_PRS_", f"_PRS{idx}_") for x in vgrd_cols]

            gefs_pres_df = pd.DataFrame(gefs_pres_u, columns=pres_ugrd_cols)
            gefs_pres_df["date"] = dates
            gefs_pres_df["hour"] = hours
            gefs_pres_df[pres_vgrd_cols] = gefs_pres_v

            if all_pressures is None:
                all_pressures = gefs_pres_df
            else:
                all_pressures = all_pressures.merge(gefs_pres_df)

        gefs_df.append(all_pressures)

    gefs_df = pd.concat(gefs_df, ignore_index=True)
    gefs_df = gefs_df.sort_values(["date", "hour"])
    gefs_df["time"] = pd.to_datetime(gefs_df.date) + pd.to_timedelta(gefs_df.hour, "h")
    gefs_df = gefs_df.set_index("time").resample("H").interpolate()
    gefs_df = gefs_df.reset_index()
    gefs_df.date = gefs_df.time.dt.date
    gefs_df.hour = gefs_df.time.dt.hour
    gefs_df = gefs_df.drop(columns="time")

    ugrd_cols = [x for x in gefs_df if "UGRD" in x]
    gefs_df = extract_ws_features(gefs_df, ugrd_cols)
    return gefs_df


def get_preds(bbox, pres_idx, scaler):
    left, right, down, up = bbox["left"], bbox["right"], bbox["down"], bbox["up"]
    pred_df = []
    vgrd_cols = [f"PRED_VGRD_PRS_LOC{i}" for i in range(4)]
    ugrd_cols = [f"PRED_UGRD_PRS_LOC{i}" for i in range(4)]
    pres_idx = [pres_idx] if isinstance(pres_idx, int) else pres_idx

    for pred_date in tqdm(os.listdir(config.test_prediction_prefix)):

        current_date = pred_date.split(".")[0]
        next_day = pd.to_datetime(current_date) + pd.to_timedelta(1, "d")
        next_day = next_day.strftime("%Y-%m-%d")

        dates = [current_date]*21 + [next_day]*3
        hours = list(range(3, 24)) + list(range(3))

        daily_pred_path = f"{config.test_prediction_prefix}/{pred_date}"
        pred = torch.load(daily_pred_path)
        pred = scaler.inverse_transform(pred, "era5")

        all_pressures = None
        for idx in pres_idx:

            pred_pres_u = pred[:, :, 2, idx, up:down+1, left:right+1].squeeze()
            pred_pres_v = pred[:,:, 3, idx, up:down+1, left:right+1].squeeze()

            pred_pres_u = pred_pres_u.view(24, 2, 2).numpy().reshape((24, 4))
            pred_pres_v = pred_pres_v.view(24, 2, 2).numpy().reshape((24, 4))

            pres_ugrd_cols = [x.replace("_PRS_", f"_PRS{idx}_") for x in ugrd_cols]
            pres_vgrd_cols = [x.replace("_PRS_", f"_PRS{idx}_") for x in vgrd_cols]

            pred_pres_df = pd.DataFrame(pred_pres_u, columns=pres_ugrd_cols)
            pred_pres_df["date"] = dates
            pred_pres_df["hour"] = hours
            pred_pres_df[pres_vgrd_cols] = pred_pres_v

            if all_pressures is None:
                all_pressures = pred_pres_df
            else:
                all_pressures = all_pressures.merge(pred_pres_df)

        pred_df.append(all_pressures)

    pred_df = pd.concat(pred_df, ignore_index=True)

    ugrd_cols = [x for x in pred_df if "UGRD" in x]
    pred_df = extract_ws_features(pred_df, ugrd_cols)

    return pred_df

def transform_bbox(row):
    lat_list = [40.5, 40.25, 40., 39.75, 39.5, 39.25, 39., 38.75, 38.5, 38.25,
       38., 37.75, 37.5, 37.25, 37., 36.75, 36.5]

    lon_list = [25., 25.25, 25.5, 25.75, 26., 26.25, 26.5, 26.75, 27., 27.25,
       27.5, 27.75, 28., 28.25, 28.5, 28.75, 29., 29.25, 29.5]

    # left right down up
    bbox_dict = {
        "left": lon_list.index(row["box_left"]),
        "right": lon_list.index(row["box_right"]),
        "down": lat_list.index(row["box_bottom"]),
        "up": lat_list.index(row["box_above"])
    }

    return bbox_dict



if __name__ == "__main__":

    if scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler(config.meta_prefix)
    elif scaler_name == "StandardScaler":
        scaler = StandardScaler(config.meta_prefix)
    else:
        raise ValueError("Scaler not valid. Options: MinMaxScaler, StandardScaler")


    eligable_farms = pd.read_csv(eligible_farm_path)

    for _, row in eligable_farms.iterrows():

        print(row)
        bbox = transform_bbox(row)
        print(bbox)
        eic = row["eic"]

        if extract_era5:
            era5_df = get_era5(bbox, era5_pres_idx)
            # Shift 3 hours for TR time
            era5_df = shift_time(era5_df, 3)
            save_path = f"{config.csv_save_path}/{eic}"
            os.makedirs(save_path, exist_ok=True)
            era5_df.to_csv(f"{save_path}/era5_pres.csv.gz", index=False)

        if extract_gefs:
            gefs_df = get_gefs(bbox, gefs_pres_idx)
            # Convert to TR time
            gefs_df = shift_time(gefs_df, 3)
            save_path = f"{config.csv_save_path}/{eic}"
            os.makedirs(save_path, exist_ok=True)
            gefs_df.to_csv(f"{save_path}/gefs_pres.csv.gz", index=False)

        pred_df = get_preds(bbox, era5_pres_idx, scaler)
        # Shift 3 hours for TR time
        pred_df = shift_time(pred_df, 3)
        save_path = f"{config.csv_save_path}/{eic}"
        os.makedirs(save_path, exist_ok=True)
        pred_df.to_csv(f"{save_path}/pred_pres.csv.gz", index=False)
