import pandas as pd
import xarray as xr
import torch

era5_prefix = "/media/harunkivril/HDD/MsThesis/ERA5"
save_prefix = "/media/harunkivril/HDD/MsThesis/ERA5_daily"
meta_save_prefix = "/media/harunkivril/HDD/MsThesis"
dim_order = ["time", "variable", "isobaricInhPa", "latitude", "longitude"]
WIN_SIZE=3
STRIDE=3
YEARS = range(2000, 2020)

temp = None
for year in YEARS:
    for month in range(1,13):
        month = str(month).zfill(2)

        if temp is None:
            temp = xr.open_dataset(f"{era5_prefix}/{year}-{month}.nc").to_array()

        start = pd.to_datetime(f"{year}-{month}-01")
        if month == "12":
            end = pd.to_datetime(f"{year+1}-01-01")
            next_file = f"{year+1}-01.nc"
        else:
            next_month = str(int(month)+1).zfill(2)
            end = pd.to_datetime(f"{year}-{next_month}-01")
            next_file = f"{year}-{next_month}.nc"

        try:
            next_temp =  xr.open_dataset(f"{era5_prefix}/{next_file}").to_array()
            temp = xr.concat([temp, next_temp], dim="time")
        except FileNotFoundError:
            next_temp = None

        temp = temp.sortby("time")
        temp = temp.drop_vars(["step", "number"])
        temp = temp.transpose(*dim_order)

        for day in pd.date_range(start, end, inclusive="left"):

            day = day + pd.to_timedelta(3, "h")
            next_day = day + pd.to_timedelta(1, "d")
            mask = (temp.time >= day) & (temp.time < next_day)
            print(day, next_day)

            if mask.sum() > 0:
                one_day = temp.where(mask, drop=True)
                one_day = one_day.drop(["d", "vo"], dim="variable")
                one_day = one_day.sortby("variable")
                to_save = one_day.values
                to_save = torch.from_numpy(to_save)
                to_save =torch.stack(
                    [
                    to_save[i:min(to_save.size(0),i + WIN_SIZE)]
                    for i in range(0, to_save.size(0), STRIDE)
                    ]
                )
                str_date = day.strftime("%Y-%m-%d")
                torch.save(to_save, f"{save_prefix}/{str_date}.pt")

        temp = next_temp

torch.save(one_day.coords, f"{meta_save_prefix}/era5_indices.pt")
