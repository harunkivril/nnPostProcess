# http://bboxfinder.com/#35.000000,25.000000,43.000000,46.000000
# http://bboxfinder.com/#35.000000,25.000000,42.500000,46.000000
# http://bboxfinder.com/#36.500000,24.500000,40.500000,29.500000
import cdsapi
import xarray as xr
import boto3
from multiprocessing import Pool

YEARS = (2016, 2015, 2014, 2013, 2012)
BUCKET = "era5-pressure"
BOUNDARIES = (36.5, 40.5, 24.5, 29.5)

def get_request_body(year, month):
    return {
                'product_type': 'reanalysis',
                'variable': [
                    'divergence', 'relative_humidity', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
                    'vorticity',
                ],
                'pressure_level': [
                    '800', '825', '850',
                    '875', '900', '925',
                    '950', '975', '1000',
                ],
                'year': f'{year}',
                'month': f'{month}',
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30', '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    43, 25, 35,
                    46,
                ],
                'format': 'grib',
            }

def request_and_pull_single_month(param):
    year, month = param

    month = str(month).zfill(2)
    grib_save_path = f'./MsThesis/NeuralPostProcess/data/{year}-{month}.grib'
    small_boundary_path = f'./NeuralPostProcess/data/{year}-{month}.nc'

    c = cdsapi.Client()
    c.retrieve(
        name='reanalysis-era5-pressure-levels',
        request=get_request_body(year, month),
        target=grib_save_path,
    )
    s3key = f"turkey/{year}-{month}.grib"
    s3cli.upload_file(grib_save_path, BUCKET, s3key)

    ds = xr.open_dataset(grib_save_path, engine='cfgrib')
    mask = (
        (ds.latitude >= BOUNDARIES[0]) &
        (ds.latitude <= BOUNDARIES[1]) &
        (ds.longitude >= BOUNDARIES[2]) &
        (ds.longitude <= BOUNDARIES[3])
    )
    ds = ds.where(mask, drop=True)
    ds.to_netcdf(small_boundary_path)
    s3key = f"aegian_region/{year}-{month}.nc"
    s3cli.upload_file(small_boundary_path, BUCKET, s3key)


if __name__ == '__main__':

    s3cli = boto3.client("s3")

    response = s3cli.list_objects_v2(
        Bucket="era5-pressure",
        Prefix="aegian_region/"
    )

    existing_files = [x["Key"].split("/")[-1] for x in response["Contents"]]
    params = (
        (year, month) for year in YEARS for month in range(1,13)
        if f'{year}-{str(month).zfill(2)}.nc' not in existing_files
    )
    pool = Pool(12)
    pool.map(request_and_pull_single_month, params)
