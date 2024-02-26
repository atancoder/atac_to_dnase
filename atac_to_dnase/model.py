import torch.nn as nn
import torch

class ATACTransformer(nn.Module):
    def __init__(self, encoding_size: int, embedding_size: int, region_width: int, num_heads: int, num_blocks: int):
        super().__init__()
        self.embedder = nn.Linear(encoding_size, embedding_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, region_width, embedding_size))  # type: ignore
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=region_width*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 1),  # predict single value (DNase signal)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedder.bias.data.zero_()
        self.embedder.weight.data.uniform_(-initrange, initrange)
        linear_unit: nn.Linear = self.decoder[0]  # type: ignore
        linear_unit.bias.data.zero_()
        linear_unit.weight.data.uniform_(-initrange, initrange)

    def forward(self, atac_encoding):
        embedding = self.embedder(atac_encoding)
        embedding += self.pos_encoder
        output = self.transformer_encoder(embedding)
        output = self.decoder(output)
        return output.squeeze()
