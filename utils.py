import torch
import os
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from fc_model import FCModel
from conv_model import ConvModel, FullyConvModel, UShapedModel

class PostProcessDataset(Dataset):

    def __init__(
        self,
        era5_daily_dir,
        gefs_daily_dir,
        transform=None,
        exclude_years=None,
        include_years=None
    ):
        super().__init__()
        self.era5_dir = era5_daily_dir
        self.gefs_dir = gefs_daily_dir
        self.transform = transform

        era5_files = set(os.listdir(self.era5_dir))
        gefs_files = set(os.listdir(self.gefs_dir))

        self.available_files = list(era5_files.intersection(gefs_files))
        self.available_files = sorted(self.available_files)

        if not exclude_years is None:
            self.available_files = [
                x for x in self.available_files
                if int(x.split("-")[0]) not in exclude_years
            ]
        elif not include_years is None:
            self.available_files = [
                x for x in self.available_files
                if int(x.split("-")[0]) in include_years
            ]

    def __len__(self):
        return len(self.available_files)*8

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        era5_path = os.path.join(self.era5_dir,
                                self.available_files[idx//8])

        gefs_path = os.path.join(self.gefs_dir,
                            self.available_files[idx//8])

        era5 = torch.load(era5_path)[idx%8]
        gefs = torch.load(gefs_path)[idx%8]

        sample = {'gefs': gefs, 'era5': era5}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MinMaxScaler:

    def __init__(self, min_max_path):

        self.era5_min, self.era5_max = torch.load(
            f"{min_max_path}/era5_min_max_vals.pt"
        )
        self.gefs_min, self.gefs_max = torch.load(
            f"{min_max_path}/gefs_min_max_vals.pt"
        )

        self.gefs_min[1:, :, :, :] = self.era5_min[1:, [0,1,2,3,4,6,8], :, :]
        self.gefs_max[1:, :, :, :] = self.era5_max[1:, [0,1,2,3,4,6,8], :, :]


    def __call__(self, sample):

        sample["gefs"] = self.transform(sample["gefs"], "gefs")
        sample["era5"] = self.transform(sample["era5"], "era5")

        return sample

    def transform(self, data, nwp_name):

        if nwp_name == "era5":
            return (
                (data - self.era5_min)/(self.era5_max - self.era5_min)
            )
        elif nwp_name == "gefs":
            return (
                (data - self.gefs_min)/(self.gefs_max - self.gefs_min)
            )
        else:
            raise ValueError("%s is not in (era5, gefs)", nwp_name)


    def inverse_transform(self, data, nwp_name):
        if nwp_name == "era5":
            return (
                data*(self.era5_max - self.era5_min) + self.era5_min
            )
        elif nwp_name == "gefs":
            return (
                data*(self.gefs_max - self.gefs_min) + self.gefs_min
            )
        else:
            raise ValueError("%s is not in (era5, gefs)", nwp_name)


class StandardScaler:

    def __init__(self, min_max_path):

        self.era5_mean, self.era5_std = torch.load(
            f"{min_max_path}/era5_mean_std_vals.pt"
        )
        self.gefs_mean, self.gefs_std = torch.load(
            f"{min_max_path}/gefs_mean_std_vals.pt"
        )

        self.gefs_mean[1:, :, :, :] = self.era5_mean[1:, [0,1,2,3,4,6,8], :, :]
        self.gefs_std[1:, :, :, :] = self.era5_std[1:, [0,1,2,3,4,6,8], :, :]


    def __call__(self, sample):

        sample["gefs"] = self.transform(sample["gefs"], "gefs")
        sample["era5"] = self.transform(sample["era5"], "era5")

        return sample

    def transform(self, data, nwp_name):

        if nwp_name == "era5":
            return (
                (data - self.era5_mean)/(self.era5_std)
            )
        elif nwp_name == "gefs":
            return (
                (data - self.gefs_mean)/(self.gefs_std)
            )
        else:
            raise ValueError("%s is not in (era5, gefs)", nwp_name)


    def inverse_transform(self, data, nwp_name):
        if nwp_name == "era5":
            return (
                data*(self.era5_std) + self.era5_mean
            )
        elif nwp_name == "gefs":
            return (
                data*(self.gefs_std) + self.gefs_mean
            )
        else:
            raise ValueError("%s is not in (era5, gefs)", nwp_name)


class PostProcessPLDataset(LightningDataModule):

    def __init__(
        self,
        batchsize,
        val_years,
        test_years,
        transform_name,
        meta_prefix,
        era5_daily_prefix,
        gefs_daily_prefix,
        dataloader_workers=10

    ):
        super().__init__()
        self.batchsize = batchsize
        self.era5_daily_prefix = era5_daily_prefix
        self.gefs_daily_prefix = gefs_daily_prefix

        if transform_name == "MinMaxScaler":
            self.transform = MinMaxScaler(meta_prefix)
        elif transform_name == "StandardScaler":
            self.transform = StandardScaler(meta_prefix)
        else:
            self.transform = None

        self.val_years = (
            [val_years] if isinstance(val_years, int) else val_years)
        self.test_years = (
            [test_years] if isinstance(test_years, int) else test_years)

        self.dataloader_workers = dataloader_workers

    def prepare_data(self):

        self.trainset = PostProcessDataset(
            era5_daily_dir=self.era5_daily_prefix,
            gefs_daily_dir=self.gefs_daily_prefix,
            transform=self.transform,
            exclude_years=self.val_years + self.test_years
        )

        self.valset = PostProcessDataset(
            era5_daily_dir=self.era5_daily_prefix,
            gefs_daily_dir=self.gefs_daily_prefix,
            transform=self.transform,
            include_years=self.val_years,
        )

        self.testset = PostProcessDataset(
            era5_daily_dir=self.era5_daily_prefix,
            gefs_daily_dir=self.gefs_daily_prefix,
            transform=self.transform,
            include_years=self.test_years,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.dataloader_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.dataloader_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=8,  # Has to be 8 to keep track of days
            shuffle=False,
            num_workers=self.dataloader_workers,
        )


def create_search_space(model_name, trial):

    if model_name == "fc_model":
        search_space = fc_model_search_space(trial)
    elif model_name == "conv_model":
        search_space = conv_model_search_space(trial)
    elif model_name == "fully_conv_model":
        search_space = fully_conv_model_search_space(trial)
    elif model_name == "ushaped_model":
        search_space = ushaped_model_search_space(trial)
    return search_space

def return_model(model_name, config):
    if model_name == "fc_model":
        model = FCModel(**config)
    elif model_name == "conv_model":
        model = ConvModel(**config)
    elif model_name == "fully_conv_model":
        model = FullyConvModel(**config)
    elif model_name == "ushaped_model":
        model = UShapedModel(**config)
    else:
        raise ValueError("Model name %s is not valid", model_name)

    return model

def return_initial_params(model_name):
    if model_name == "fc_model":
        params = {
            "hidden_size": 1024,
            "num_layers": 1,
            "learning_rate": 1e-3,
            "batchsize": 32,
            "weight_decay": 0,
            "dropout": 0,
            "use_batchnorm": True
        }
    elif model_name == "conv_model":
        params = {
            "num_channels": 256,
            "num_conv_layers": 3,
            "num_layers": 1,
            "hidden_size": 1024,
            "learning_rate": 1e-3,
            "batchsize": 32,
            "weight_decay": 0,
            "dropout": 0,
            "use_batchnorm": True,
            "use_ndbatchnorm": True,
            "pooling_func_name": "AvgPool",
        }
    elif model_name == "fully_conv_model":
        params = {
            "num_channels": 256,
            "num_layers": 4,
            "learning_rate": 1e-3,
            "batchsize": 32,
            "weight_decay": 0,
            "dropout": 0,
            "use_ndbatchnorm": True,
            "pooling_func_name": "AvgPool",
        }
    elif model_name == "ushaped_model":
        params = {
            "num_channels": 256,
            "learning_rate": 1e-3,
            "batchsize": 32,
            "weight_decay": 0,
            "dropout": 0,
            "use_ndbatchnorm": True,
            "pooling_func_name": "AvgPool",
        }
    else:
        params = {}

    return params

def fc_model_search_space(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 1024]),
    num_layers = trial.suggest_int("num_layers", 1, 5)
    space = {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batchsize": trial.suggest_categorical("batchsize", [8, 16, 32, 64, 128, 256]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-2]),
        "fc_outs": [hidden_size]*num_layers,
        "dropout": trial.suggest_float("dropout", 0, 0.5, step=0.1),
        "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False])

    }
    return space

def conv_model_search_space(trial):
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 6)
    num_channels = trial.suggest_categorical("num_channels", [64, 128, 256, 512, 1024])
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 1024])
    num_layers = trial.suggest_int("num_layers", 1, 3)

    space = {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batchsize": trial.suggest_categorical("batchsize", [8, 16, 32, 64, 128, 256]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-2]),
        "fc_outs": [hidden_size]*num_layers,
        "channel_outs": [num_channels]*num_conv_layers,
        "dropout": trial.suggest_float("dropout", 0, 0.5, step=0.1),
        "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
        "use_ndbatchnorm": trial.suggest_categorical("use_ndbatchnorm", [True, False]),
        "pooling_func_name": trial.suggest_categorical("pooling_func_name", ["AvgPool", "MaxPool"]),

    }
    return space

def fully_conv_model_search_space(trial):
    num_conv_layers = trial.suggest_int("num_layers", 1, 8)
    num_channels = trial.suggest_categorical("num_channels", [64, 128, 256, 512, 1024])
    space = {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batchsize": trial.suggest_categorical("batchsize", [8, 16, 32, 64, 128, 256]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-2]),
        "channel_outs": [num_channels]*num_conv_layers,
        "dropout": trial.suggest_float("dropout", 0, 0.5, step=0.1),
        "use_ndbatchnorm": trial.suggest_categorical("use_ndbatchnorm", [True, False]),
        "pooling_func_name": trial.suggest_categorical("pooling_func_name", ["AvgPool", "MaxPool"]),
    }
    return space

def ushaped_model_search_space(trial):
    num_channels = trial.suggest_categorical("num_channels", [64, 128, 256, 512, 1024])
    space = {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batchsize": trial.suggest_categorical("batchsize", [8, 16, 32, 64, 128, 256]),
        "weight_decay": trial.suggest_categorical("weight_decay", [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-2]),
        "channel_outs": [num_channels]*8,
        "dropout": trial.suggest_float("dropout", 0, 0.5, step=0.1),
        "use_ndbatchnorm": trial.suggest_categorical("use_ndbatchnorm", [True, False]),
        "pooling_func_name": trial.suggest_categorical("pooling_func_name", ["AvgPool", "MaxPool"]),
    }
    return space

def return_time_idx(hour):
    if hour < 3:
        print("WARNING: The value for the given day is in prev tensor. Returning next day")
        return 7, hour

    first_dim = hour // 3 -1
    second_dim = hour % 3

    return first_dim, second_dim

def filter_era5_tensor(tensor, hour, variable, level, lat, lon):
    variable_list = ['r', 't', 'u', 'v', 'w']
    level_list = [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.]
    lat_list = [40.5, 40.25, 40., 39.75, 39.5, 39.25, 39., 38.75, 38.5, 38.25,
       38., 37.75, 37.5, 37.25, 37., 36.75, 36.5]
    lon_list = [25., 25.25, 25.5, 25.75, 26., 26.25, 26.5, 26.75, 27., 27.25,
       27.5, 27.75, 28., 28.25, 28.5, 28.75, 29., 29.25, 29.5]

    var_idx = variable_list.index(variable)
    level_idx = level_list.index(level)
    lat_idx = lat_list.index(lat)
    lon_idx = lon_list.index(lon)
    time1, time2 = return_time_idx(hour)

    print(f"Indices: {time1}, {time2}, {var_idx}, {level_idx}, {lat_idx}, {lon_idx}")

    return tensor[time1, time2, var_idx, level_idx, lat_idx, lon_idx]

def filter_gefs_tensor(tensor, hour, variable, level, lat, lon):
    variable_list = ['q', 't', 'u', 'v', 'w']
    level_list = [1000,  975,  950,  925,  900,  850,  800]
    lat_list = [40.5, 40.25, 40., 39.75, 39.5, 39.25, 39., 38.75, 38.5, 38.25,
       38., 37.75, 37.5, 37.25, 37., 36.75, 36.5]
    lon_list = [25., 25.25, 25.5, 25.75, 26., 26.25, 26.5, 26.75, 27., 27.25,
       27.5, 27.75, 28., 28.25, 28.5, 28.75, 29., 29.25, 29.5]

    var_idx = variable_list.index(variable)
    level_idx = level_list.index(level)
    lat_idx = lat_list.index(lat)
    lon_idx = lon_list.index(lon)

    assert hour % 3 == 0
    time_idx, _ = return_time_idx(hour)

    print(f"Indices: {time_idx}, {var_idx}, {level_idx}, {lat_idx}, {lon_idx}")

    return tensor[time_idx, var_idx, level_idx, lat_idx, lon_idx]
