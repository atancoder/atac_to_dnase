from sympy import content, sequence
from torch import nn
import torch
import torch.nn.functional as F


class RelativePositionEmbedding(nn.Module):
    def __init__(self, max_seq_length, model_dim):
        super(RelativePositionEmbedding, self).__init__()
        self.rel_pos_embedding = nn.Embedding(2 * max_seq_length - 1, model_dim)
        # Generate a matrix of relative positions
        positions = torch.arange(max_seq_length).unsqueeze(0) - torch.arange(max_seq_length).unsqueeze(1)
        # Shift positions to ensure all indices are positive
        self.shifted_positions = positions + max_seq_length - 1

    def forward(self, device):
        # Retrieve the embeddings for these relative positions
        return self.rel_pos_embedding(self.shifted_positions.to(device))

class MultiHeadAttentionWithRelativePosition(nn.Module):
    def __init__(self, channels, num_heads, max_seq_length):
        super(MultiHeadAttentionWithRelativePosition, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_size = channels // num_heads

        self.Q = nn.Linear(channels, channels)
        self.K = nn.Linear(channels, channels)
        self.V = nn.Linear(channels, channels)
        self.relative_position_k = RelativePositionEmbedding(max_seq_length, channels//num_heads)
        self.relative_position_v = RelativePositionEmbedding(max_seq_length, channels//num_heads)

    def forward(self, src):
        batch_size = src.size(0)
        seq_length = src.size(1)

        # partitions the vectors into num_heads
        q = self.Q(src).view(batch_size, self.num_heads, seq_length, self.head_size) 
        k = self.K(src).view(batch_size, self.num_heads, seq_length, self.head_size)
        v = self.V(src).view(batch_size, self.num_heads, seq_length, self.head_size)
        
        # Leads to shape = (batch_size, num_heads, seq_length, seq_length)
        content_score = torch.matmul(q, k.permute(0, 1, 3, 2))

        # shape = (seq_length, seq_length, head_size)
        # But need to change to (seq_length, head_size, seq_length)
        rel_k = self.relative_position_k(src.device).permute(0, 2, 1)
        q_reshaped = q.view(seq_length, batch_size*self.num_heads, self.head_size)

        # shape = (seq_length, batch*num_heads, seq_length)
        position_score = torch.matmul(q_reshaped, rel_k)
        position_score = position_score.permute(1,0,2).view(batch_size, self.num_heads, seq_length, seq_length)

        scores = (content_score + position_score) / (self.head_size ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # shape = (batch_size, num_heads, seq_length, head_size)
        attn_output_vals = torch.matmul(attention, v)

        # Concatenate heads
        # shape = (batch_size, seq_length, num_heads, head_size)
        attn_output_vals = attn_output_vals.permute(0, 2, 1, 3).contiguous()
        output = attn_output_vals.view(batch_size, seq_length, self.channels)
        return output

class TransformerEncoderLayerWithRelativePosition(nn.Module):
    def __init__(self, channels, num_heads, max_seq_length):
        assert channels % num_heads == 0, "Channels must be divisible by num heads"
        super(TransformerEncoderLayerWithRelativePosition, self).__init__()
        self.multi_head_attention = MultiHeadAttentionWithRelativePosition(channels, num_heads, max_seq_length)
        self.layer_norm1 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.ReLU(),
            nn.Linear(4 * channels, channels)
        )
        self.layer_norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        attn = self.multi_head_attention(src)
        ln1_out = self.layer_norm1(src + self.dropout(attn))
        
        ff_out = self.feed_forward(ln1_out)
        output = self.layer_norm2(ln1_out + self.dropout(ff_out))
        return output
