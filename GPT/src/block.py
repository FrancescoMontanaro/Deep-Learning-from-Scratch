import torch
import torch.nn as nn

from .config import GPTConfig
from .mlp import MultiLayerPerceptron
from .attention_mechanism import MultiHeadAttention


class Block(nn.Module):
    
    ### Magic Methods ###
    
    def __init__(self, config: GPTConfig) -> None:
        """
        Class constructor
        
        Parameters:
        - config (GPTConfig): The configuration class for the GPT model.
        """
        
        # Initialize the parent class
        super().__init__()
        
        # Creating the block layers
        self.layer_norm_1 = nn.LayerNorm(config.n_embed) # First normalization layer
        self.attention = MultiHeadAttention(config) # Multi-head attention mechanism
        self.layer_norm_2 = nn.LayerNorm(config.n_embed) # Second normalization layer
        self.mlp = MultiLayerPerceptron(config) # Multi-layer perceptron layer
        
        
    ### Public Methods ###
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the block.
        
        Parameters:
        - x (torch.Tensor): The input embeddings.
        
        Returns:
        - torch.Tensor: The output embeddings.
        """
        
        # Apply the attention mechanism to the normalized input embeddings
        x = x + self.attention(self.layer_norm_1(x))
        
        # Apply the multi-layer perceptron to the normalized input embeddings
        x = x + self.mlp(self.layer_norm_2(x))
        
        # Return the output embeddings
        return x  