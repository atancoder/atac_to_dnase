import torch.nn as nn
import torch

class ATACTransformer(nn.Module):
    def __init__(self, n_encoding: int, channels: int, region_width: int, num_heads: int, num_blocks: int):
        super().__init__()
        self.embedder = nn.Linear(n_encoding, channels)
        self.pos_encoder = nn.Parameter(torch.zeros(1, region_width, channels))  # type: ignore
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=num_heads, dim_feedforward=region_width*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks)
        self.decoder = nn.Sequential(
            nn.Linear(channels, 1),  # predict single value (DNase signal)
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of Linear layers using He initialization
        """
        nn.init.kaiming_uniform_(self.embedder.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_uniform_(self.decoder[0].weight, mode='fan_in', nonlinearity='linear')

    def forward(self, encoding):
        embedding = self.embedder(encoding)
        embedding = embedding + self.pos_encoder
        output = self.transformer_encoder(embedding)
        output = self.decoder(output)
        return output.squeeze()
