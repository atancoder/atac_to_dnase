from typing import Any, Callable
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import relu
import torch
from torch.nn import functional as F

class ATACTransformer(nn.Module):
    def __init__(self, encoding_size: int, region_width: int, num_heads: int, num_blocks: int):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(1, region_width, encoding_size))  # type: ignore
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=encoding_size, nhead=num_heads, dim_feedforward=region_width*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks)
        self.encoding_size = encoding_size
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 1),  # predict single value (DNase signal)
            nn.ReLU()
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        linear_unit: nn.Linear = self.decoder[0]  # type: ignore
        linear_unit.bias.data.zero_()
        linear_unit.weight.data.uniform_(-initrange, initrange)

    def forward(self, atac_encoding):
        atac_encoding += self.pos_encoder
        output = self.transformer_encoder(atac_encoding)
        output = self.decoder(output)
        return output.squeeze()
