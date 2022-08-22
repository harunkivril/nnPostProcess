import torch.nn as nn
import torch
from modelbase import ModelBase
from model_utils import Conv2dBlock, Conv3dBlock, FCBlock

class ConvModel(ModelBase):
    def __init__(
        self,
        conv_dim,
        kernel_size,
        stride,
        pad,
        channel_outs,
        fc_outs,
        dropout,
        use_batchnorm,
        use_ndbatchnorm,
        pooling_func_name,
        optimizer_name,
        learning_rate,
        weight_decay,
        test_prediction_prefix,
        test_start_year,
        loss_function_name,
        name="conv_model",
        **kwargs
    ):

        super().__init__(optimizer_name, learning_rate, weight_decay,
        test_prediction_prefix, test_start_year, loss_function_name, **kwargs)


        self.fc_outs = list(fc_outs)
        self.channel_outs = list(channel_outs)

        if conv_dim == 2:
            self.in_channel = self.gefs_dims[0] * self.gefs_dims[1]
            self.view_size = self.in_channel, self.gefs_dims[-2], self.gefs_dims[-1]
            block_class = Conv2dBlock
        elif conv_dim == 3:
            self.in_channel = self.gefs_dims[0]
            self.view_size = (self.in_channel, self.gefs_dims[-3], self.gefs_dims[-2], self.gefs_dims[-1])
            block_class = Conv3dBlock
        else:
            raise ValueError("Conv dim %s is not valid. Options 2, 3", conv_dim)

        self.conv_blocks = nn.ModuleList([])
        self.fc_blocks = nn.ModuleList([])
        for i, out_ch in enumerate(self.channel_outs):
            in_ch = channel_outs[i-1] if i > 0 else self.in_channel
            block = block_class(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                pool_type=pooling_func_name,
                batch_norm=use_ndbatchnorm,
                dropout=dropout,
            )
            self.conv_blocks.append(block)

        # Set fully connected layers
        psudo_batch = 1
        input_size = [psudo_batch] + list(self.gefs_dims)
        fc_in_dim = self.conv_forward(
            x=torch.rand(size=input_size),
            batch_size=psudo_batch
        ).view(psudo_batch, -1).shape[1]

        for i, fc_out in enumerate(self.fc_outs):
            fc_in = fc_outs[i-1] if i > 0 else fc_in_dim
            block = FCBlock(fc_in, fc_out, use_batchnorm, dropout)
            self.fc_blocks.append(block)
        self.out_layer = nn.Linear(fc_out, self.out_dim)

    def conv_forward(self, x, batch_size):
        x = x.view(batch_size, *self.view_size)
        for block in self.conv_blocks:
            x = block(x)
        return x

    def fc_forward(self, x, batch_size):
        x = x.view(batch_size, -1)
        for block in self.fc_blocks:
            x = block(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_forward(x, batch_size)
        x = self.fc_forward(x, batch_size)
        x = self.out_layer(x)
        x = x.view(batch_size, *self.era5_dims)
        return x

class FullyConvModel(ModelBase):
    def __init__(
        self,
        conv_dim,
        channel_outs,
        pooling_func_name,
        dropout,
        optimizer_name,
        learning_rate,
        weight_decay,
        use_ndbatchnorm,
        test_prediction_prefix,
        test_start_year,
        loss_function_name,
        name="fully_conv_model",
        **kwargs
    ):
        super().__init__(optimizer_name, learning_rate, weight_decay,
            test_prediction_prefix, test_start_year, loss_function_name, **kwargs)

        if conv_dim == 2:
            self.in_channel = self.gefs_dims[0] * self.gefs_dims[1]
            self.view_size = self.in_channel, self.gefs_dims[-2], self.gefs_dims[-1]
            self.out_channel = self.era5_dims[0] * self.era5_dims[1] * self.era5_dims[2]
            block_class = Conv2dBlock
        elif conv_dim == 3:
            self.in_channel = self.gefs_dims[1]
            self.view_size = (self.in_channel, self.gefs_dims[-4], self.gefs_dims[-2], self.gefs_dims[-1])
            self.out_channel = self.era5_dims[0] * self.era5_dims[2]  #Add missing two pressure levels
            block_class = Conv3dBlock
        else:
            raise ValueError("Conv dim %s is not valid. Options 2, 3", conv_dim)

        self.channel_outs = list(channel_outs) + [self.out_channel]

        self.conv_blocks = nn.ModuleList([])
        for i, out_ch in enumerate(self.channel_outs):
            in_ch = channel_outs[i-1] if i > 0 else self.in_channel
            block = block_class(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=2,
                stride=1,
                padding=1,
                pool_type=pooling_func_name,
                batch_norm=use_ndbatchnorm,
                dropout=dropout,
            )
            self.conv_blocks.append(block)

    def conv_forward(self, x, batch_size):
        x = x.view(batch_size, *self.view_size)
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_forward(x, batch_size)
        x = x.view(batch_size, *self.era5_dims)
        return x


class FullyConvResModel(ModelBase):
    def __init__(
        self,
        conv_dim,
        channel_outs,
        pooling_func_name,
        dropout,
        optimizer_name,
        learning_rate,
        weight_decay,
        use_ndbatchnorm,
        test_prediction_prefix,
        test_start_year,
        loss_function_name,
        name = "fully_conv_res_model",
        **kwargs
    ):
        super().__init__(optimizer_name, learning_rate, weight_decay,
            test_prediction_prefix, test_start_year, loss_function_name, **kwargs)

        if conv_dim == 2:
            self.in_channel = self.gefs_dims[0] * self.gefs_dims[1]
            self.view_size = self.in_channel, self.gefs_dims[-2], self.gefs_dims[-1]
            self.out_channel = self.era5_dims[0] * self.era5_dims[1] * self.era5_dims[2]
            block_class = Conv2dBlock
        elif conv_dim == 3:
            self.in_channel = self.gefs_dims[1]
            self.view_size = (self.in_channel, self.gefs_dims[-4], self.gefs_dims[-2], self.gefs_dims[-1])
            self.out_channel = self.era5_dims[0] * self.era5_dims[2]  #Add missing two pressure levels
            block_class = Conv3dBlock
        else:
            raise ValueError("Conv dim %s is not valid. Options 2, 3", conv_dim)

        self.channel_outs = list(channel_outs)
        # Add extra channels for residual connection
        self.last_block_channel_in = self.channel_outs[-1] + self.in_channel

        self.conv_blocks = nn.ModuleList([])
        for i, out_ch in enumerate(self.channel_outs):
            in_ch = channel_outs[i-1] if i > 0 else self.in_channel
            block = block_class(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=2,
                stride=1,
                padding=1,
                pool_type=pooling_func_name,
                batch_norm=use_ndbatchnorm,
                dropout=dropout,
            )
            self.conv_blocks.append(block)

        self.last_block = block_class(
            in_ch=self.last_block_channel_in,
            out_ch=self.out_channel,
            kernel_size=2,
            stride=1,
            padding=1,
            pool_type=pooling_func_name,
            batch_norm=use_ndbatchnorm,
            dropout=dropout,
        )

    def conv_forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, *self.view_size)
        out = self.conv_forward(x)
        x = torch.concat([out, x], dim=1)
        x = self.last_block(x)
        x = x.view(batch_size, *self.era5_dims)
        return x

class UShapedModel(ModelBase):
    def __init__(
        self,
        conv_dim,
        channel_outs,
        pooling_func_name,
        dropout,
        optimizer_name,
        learning_rate,
        weight_decay,
        use_ndbatchnorm,
        test_prediction_prefix,
        test_start_year,
        loss_function_name,
        name="ushaped_model",
        **kwargs
    ):
        super().__init__(optimizer_name, learning_rate, weight_decay,
            test_prediction_prefix, test_start_year, loss_function_name, **kwargs)

        if conv_dim == 2:
            self.in_channel = self.gefs_dims[0] * self.gefs_dims[1]
            self.view_size = self.in_channel, self.gefs_dims[-2], self.gefs_dims[-1]
            self.out_channel = self.era5_dims[0] * self.era5_dims[1] * self.era5_dims[2]
            block_class = Conv2dBlock
        elif conv_dim == 3:
            self.in_channel = self.gefs_dims[1]
            self.view_size = (self.in_channel, self.gefs_dims[-4], self.gefs_dims[-2], self.gefs_dims[-1])
            self.out_channel = self.era5_dims[0] * self.era5_dims[2]  #Add missing two pressure levels
            block_class = Conv3dBlock
        else:
            raise ValueError("Conv dim %s is not valid. Options 2, 3", conv_dim)

        self.channel_outs = list(channel_outs) + [self.out_channel]
        assert len(channel_outs) == 8
        # Add extra channels for residual connection

        self.conv_blocks = nn.ModuleList([])
        for i, out_ch in enumerate(self.channel_outs):
            in_ch = channel_outs[i-1] if i > 0 else self.in_channel
            in_ch = in_ch + channel_outs[-i] if i > 4 else in_ch
            block = block_class(
                in_ch=in_ch,
                out_ch=out_ch,
                kernel_size=2,
                stride=1,
                padding=1,
                pool_type=pooling_func_name,
                batch_norm=use_ndbatchnorm,
                dropout=dropout,
            )
            self.conv_blocks.append(block)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, *self.view_size)

        outs = []
        for block in self.conv_blocks[:4]:
            x = block(x)
            outs.append(x)

        x = self.conv_blocks[4](x)

        for i, block in enumerate(self.conv_blocks[5:]):
            x = torch.concat([outs[-i], x], dim=1)
            x = block(x)

        x = x.view(batch_size, *self.era5_dims)
        return x