import os
import xarray as xr
import boto3
import pandas as pd

def clear_directory(path):
    for file in os.listdir(path):
        os.remove(path + file)


def process_dates(dates):
    final_ds = []
    for date in dates:
        print(date)
        var_ds = None
        for var in VARS:
            key = f"GEFSv12/reforecast/{date[:4]}/{date}00/c00/Days:1-10/{var}_pres_{date}00_c00.grib2"
            grib_path = GRIB_SAVE_PATH + key.split("/")[-1]
            response = s3cli.download_file(
                Bucket=BUCKET,
                Key=key,
                Filename=grib_path
            )
            ds = xr.open_dataset(grib_path, engine='cfgrib')

            turkey_pres_mask = (
                    (ds.latitude >= TURKEY_BOUNDARIES[0]) &
                    (ds.latitude <= TURKEY_BOUNDARIES[1]) &
                    (ds.longitude >= TURKEY_BOUNDARIES[2]) &
                    (ds.longitude <= TURKEY_BOUNDARIES[3]) &
                    (ds.isobaricInhPa  >= PRES_BOUNDARIES[0]) &
                    (ds.isobaricInhPa  <= PRES_BOUNDARIES[1]) &
                    (ds.step <= MAX_STEP)
            )

            ds = ds.where(turkey_pres_mask, drop=True)
            if var_ds is None:
                var_ds = ds
            else:
                var_ds = var_ds.merge(ds)

        clear_directory(GRIB_SAVE_PATH)
        final_ds.append(var_ds)

    final_ds = xr.concat(final_ds, dim="time")
    return final_ds

if __name__ == '__main__':
    YEARS = (2017, 2016)
    BUCKET = "noaa-gefs-retrospective"
    TARGET_BUCKET = "gefs-pressure"
    TURKEY_BOUNDARIES =  (35, 43, 25, 46)
    AEGIAN_BOUNDARIES = (36.5, 40.5, 24.5, 29.5)
    PRES_BOUNDARIES = (800, 1000)
    MAX_STEP = pd.to_timedelta(72, "h")
    SAVE_PREFIX = "./data/GEFS/"
    GRIB_SAVE_PATH = SAVE_PREFIX + "full_gribs/"

    VARS = ("ugrd", "vgrd", "tmp", "vvel", "spfh")
    s3cli = boto3.client("s3")

    os.makedirs(GRIB_SAVE_PATH, exist_ok=True)

    response = s3cli.list_objects_v2(
            Bucket=TARGET_BUCKET,
            Prefix="aegian_region/"
    )

    existing_files = [x["Key"].split("/")[-1] for x in response.get("Contents", ())]

    for year in YEARS:
        year_days = pd.date_range(f"{year}0101", f"{year}1231")
        for month in range(1, 13):
            print("Year month:",year, month)
            if f"{year}-{month}.nc" in existing_files:
                continue
            month_days = [x.strftime("%Y%m%d") for x in year_days if x.month == month]
            month = str(month).zfill(2)
            month_data = process_dates(month_days)

            turkey_nc_path = SAVE_PREFIX + f"/{year}-{month}.nc"
            month_data.to_netcdf(turkey_nc_path)
            s3key = f"turkey/{year}-{month}.nc"
            s3cli.upload_file(turkey_nc_path, TARGET_BUCKET, s3key)

            aegian_mask = (
                (month_data.latitude >= AEGIAN_BOUNDARIES[0]) &
                (month_data.latitude <= AEGIAN_BOUNDARIES[1]) &
                (month_data.longitude >= AEGIAN_BOUNDARIES[2]) &
                (month_data.longitude <= AEGIAN_BOUNDARIES[3])
            )

            aegian_data = month_data.where(aegian_mask, drop=True)
            aegian_nc_path = SAVE_PREFIX +f"{year}-{month}_aegian.nc"
            aegian_data.to_netcdf(aegian_nc_path)
            s3key = f"aegian_region/{year}-{month}.nc"
            s3cli.upload_file(aegian_nc_path, TARGET_BUCKET, s3key)
