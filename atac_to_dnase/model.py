import torch.nn as nn
import torch

class ATACTransformer(nn.Module):
    def __init__(self, vocab_size: int, channels: int, region_width: int, num_heads: int, num_blocks: int):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, channels)
        embedding_size = channels + 1  # we will concat atac signal
        self.linear_unit = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU()
        )
        self.pos_encoder = nn.Parameter(torch.zeros(1, region_width, embedding_size))  # type: ignore
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=num_heads, dim_feedforward=region_width*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 1),  # predict single value (DNase signal)
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of Linear layers using He initialization
        """
        nn.init.kaiming_uniform_(self.linear_unit[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder[0].weight, mode='fan_in', nonlinearity='linear')

    def forward(self, encoding):
        seq_encoding = encoding[:,:,0].to(torch.int32)
        atac_encoding = encoding[:,:,1]
        seq_embedding = self.embedding_table(seq_encoding)
        embedding = torch.concat((seq_embedding, atac_encoding.unsqueeze(-1)), dim=-1)
        embedding = self.linear_unit(embedding)
        embedding = embedding + self.pos_encoder
        output = self.transformer_encoder(embedding)
        output = self.decoder(output)
        return output.squeeze()
