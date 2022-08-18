import pandas as pd
import xarray as xr
import torch

gefs_prefix = "/media/harunkivril/HDD/MsThesis/GEFS"
save_prefix = "/media/harunkivril/HDD/MsThesis/GEFS_daily"
meta_save_prefix = "/media/harunkivril/HDD/MsThesis"
dim_order = ["time", "step", "variable", "isobaricInhPa", "latitude", "longitude"]
YEARS = range(2000, 2020)
max_ahead = pd.to_timedelta(24, "h")

for year in YEARS:
    for month in range(1,13):

        month = str(month).zfill(2)
        temp = xr.open_dataset(f"{gefs_prefix}/{year}-{month}.nc").to_array()

        temp = temp.sortby("time")
        temp = temp.drop_vars(["number"])
        temp = temp.transpose(*dim_order)

        start = pd.to_datetime(f"{year}-{month}-01")
        if month == "12":
            end = pd.to_datetime(f"{year+1}-01-01")
        else:
            next_month = str(int(month)+1).zfill(2)
            end = pd.to_datetime(f"{year}-{next_month}-01")

        for day in pd.date_range(start, end, inclusive="left"):

            next_day = day + pd.to_timedelta(1, "d")
            mask = (temp.time >= day) & (temp.time < next_day) & (temp.step <= max_ahead)
            print(day, next_day)

            if mask.sum() > 0:
                one_day = temp.where(mask, drop=True)
                one_day = one_day.sortby("variable")
                to_save = one_day.values[0]
                to_save = torch.from_numpy(to_save)
                str_date = day.strftime("%Y-%m-%d")
                torch.save(to_save, f"{save_prefix}/{str_date}.pt")

torch.save(dict(one_day.coords), f"{meta_save_prefix}/gefs_indices.pt")

