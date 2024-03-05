import torch.nn as nn
import torch
from .utils import ONE_HOT_ENCODING_SIZE

NUM_HEADS = 8
NUM_BLOCKS = 8
CONV_FILTER_SIZE = 21  # Needs to be odd
CHANNELS = 64

class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input) + input
    
class CenterRegion(nn.Module):
    """
    When applying convolutions, we have shrunk the context length. When 
    considering an element to make a prediction on, we want to choose the 
    element after the convolution where it has context from the left and the right.

    For example, say filter size was 11. And we had context length of 20. One of the 
    filters will capture [e1,e2,e3,..,e11]. We want this capture to represent e6, as
    e6 here has context from both sides of its neighbors 
    """
    
    def __init__(self, region_width: int, region_slop: int) -> None:
        super().__init__()
        self.region_width = region_width
        self.region_slop = region_slop
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Ideally, we want elements from [region_slop: -region_slop], but the
        elements we want have been shifted due to the convolution. So we
        need to adjust for that offset
        """
        element_index_offset = -1 * (CONV_FILTER_SIZE//2)
        start_idx = self.region_slop + element_index_offset
        end_idx = (self.region_width - self.region_slop) + element_index_offset
        return input[:,start_idx: end_idx,:]

class ATACTransformer(nn.Module):
    def __init__(self, region_width: int, region_slop: int):
        assert CONV_FILTER_SIZE % 2 == 1
        super().__init__()
        encoding_size = ONE_HOT_ENCODING_SIZE + 1  # ATAC signal concatenated 
        embedding_conv_layer = nn.Sequential(
            nn.Conv1d(encoding_size, CHANNELS, CONV_FILTER_SIZE, stride=1, padding='valid'),
            nn.BatchNorm1d(CHANNELS),
            nn.GELU()
        )
        new_width = region_width - CONV_FILTER_SIZE + 1
        residual_conv_layer = Residual(nn.Conv1d(CHANNELS, CHANNELS, 1, 1))
        self.conv_layers = nn.Sequential(
            embedding_conv_layer, residual_conv_layer
        )  # We only need 1 conv layer as attention would handle neighbor interactions

        # Absolute encodings for now
        self.pos_encoder = nn.Parameter(torch.zeros(1, new_width, CHANNELS))  # type: ignore

        encoder_layer = nn.TransformerEncoderLayer(d_model=CHANNELS, nhead=NUM_HEADS, dim_feedforward=new_width*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_BLOCKS)
        self.crop = CenterRegion(region_width, region_slop)
        self.decoder = nn.Sequential(
            nn.Linear(CHANNELS, 1),
            nn.Softplus()  # Using ReLu leads to dying neurons early on
        )
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of Linear layers using He initialization
        """
        nn.init.kaiming_uniform_(self.decoder[0].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, dna_encoding: torch.Tensor, atac_signal: torch.Tensor) -> torch.Tensor:
        # dna_encoding.shape = B, T, C
        # atac_signal.shape = B, T
        encoding = torch.cat((dna_encoding, atac_signal.unsqueeze(-1)), dim=2)
        encoding = encoding.permute(0, 2, 1)  # Need to change dim for conv layers
        embedding = self.conv_layers(encoding)
        embedding = embedding.permute(0, 2, 1)

        embedding = embedding + self.pos_encoder
        hidden_states = self.transformer_encoder(embedding)
        hidden_states = self.crop(hidden_states)
        dnase_signal = self.decoder(hidden_states)
        return dnase_signal
