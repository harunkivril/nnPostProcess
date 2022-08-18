import torch.nn as nn
from modelbase import ModelBase
from model_utils import FCBlock

class FCModel(ModelBase):
    def __init__(
        self,
        fc_outs,
        dropout,
        use_batchnorm,
        optimizer_name,
        learning_rate,
        weight_decay,
        test_prediction_prefix,
        test_start_year,
        loss_function_name,
        name="fc_model",
        **kwargs
    ):

        super().__init__(optimizer_name, learning_rate, weight_decay,
        test_prediction_prefix, test_start_year, loss_function_name, **kwargs)

        self.fc_outs = list(fc_outs)
        self.fc_blocks = nn.ModuleList([])
        for i, fc_out in enumerate(self.fc_outs):
            fc_in = fc_outs[i-1] if i > 0 else self.in_dim
            block = FCBlock(fc_in, fc_out, use_batchnorm, dropout)
            self.fc_blocks.append(block)
        self.out_layer = nn.Linear(fc_out, self.out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        for block in self.fc_blocks:
            x = block(x)

        x = self.out_layer(x)
        x = x.view(batch_size, *self.era5_dims)

        return x
