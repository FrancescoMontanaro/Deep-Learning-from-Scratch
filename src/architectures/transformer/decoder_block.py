import torch
import torch.nn as nn

from .mlp import MLP
from .attention_mechanism import MultiHeadAttention


class DecoderBlock(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float = 0.1) -> None:
        """
        Initialize the transformer's decoder block.
        
        Parameters:
        - n_embed (int): The size of the embeddings.
        - n_heads (int): The number of attention heads.
        - block_size (int): The size of the block.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the superclass
        super().__init__()
        
        # Compute the size of the attention heads by dividing the embedding size by the number of heads
        head_size = n_embed // n_heads
        
        # Create the multi-head self-attention mechanism and the feed-forward layers
        self.self_attention_heads = MultiHeadAttention(  # Create the multi-head self-attention mechanism
            n_heads = n_heads, 
            head_size = head_size, 
            n_embed = n_embed, 
            block_size = block_size,
            dropout = dropout
        )
        self.feed_forward = MLP(  # Create the feed-forward layers
            n_embed = n_embed,
            dropout = dropout
        )
        self.layer_norm_1 = nn.LayerNorm(n_embed) # Create the normalization layer of the attention mechanism
        self.layer_norm_2 = nn.LayerNorm(n_embed) # Create the normalization layer of the feed-forward layers
      
    
    ### Public methods ###  
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer block.
        
        Parameters:
        - embeddings (torch.Tensor): The input embeddings.
        
        Returns:
        - torch.Tensor: The output embeddings.
        """
        
        # Apply the self-attention mechanism with skip connections
        embeddings = embeddings + self.self_attention_heads(self.layer_norm_1(embeddings))
        
        # Apply the feed-forward layers with skip connections
        embeddings = embeddings + self.feed_forward(self.layer_norm_2(embeddings))
        
        return embeddings
