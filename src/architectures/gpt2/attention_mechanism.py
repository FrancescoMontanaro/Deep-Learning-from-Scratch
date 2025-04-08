import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class MultiHeadAttention(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, config: GPTConfig) -> None:
        """
        Class constructor
        
        Parameters:
        - config (GPTConfig): The configuration class for the GPT model.
        
        Raises:
        - AssertionError: If the size of the embeddings is not divisible by the number of heads.
        """
        
        # Initialize the parent class
        super().__init__()
        
        # Asserting that the size of the embeddings is divisible by the number of heads
        assert config.n_embed % config.n_heads == 0, "The size of the embeddings must be divisible by the number of heads."
        
        # Storing the number of heads and the size of the embeddings
        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        
        # Creating an unique tensor for keys, queries, and values
        self.atten_kqv = nn.Linear(config.n_embed, 3 * config.n_embed)
        
        # Creating the output linear layer
        self.output_linear = nn.Linear(config.n_embed, config.n_embed)
        self.output_linear.residual_layer = 1 # Add a flag to the output linear layer to indicate that it is a residual layer # type: ignore
        
        
    ### Public methods ###
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.
        
        Parameters:
        - x (torch.Tensor): The input embeddings.
        
        Returns:
        - torch.Tensor: The output embeddings.
        """
        
        # Extract the shape of the input tensor
        B, T, C = x.size() # Batch size, sequence length, number of embeddings (n_embed)
        
        # Compute key, query, and value matrices for each head in a single linear transformation
        k, q, v = self.atten_kqv(x).chunk(3, dim=-1)
        
        # Reshape the key, query, and value matrices to have the same shape as the input tensor
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # Batch size, number of heads, sequence length, head size
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # Batch size, number of heads, sequence length, head size
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # Batch size, number of heads, sequence length, head size
        
        # The attention mechanism is implemented using the scaled dot-product attention function
        # which in turn implements the flash attention mechanism (https://arxiv.org/abs/2307.08691)
        contextual_embeddings = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Batch size, number of heads, sequence length, head size
        
        # Reshape the contextual embeddings to have the same shape as the input tensor
        contextual_embeddings = contextual_embeddings.transpose(1, 2).contiguous().view(B, T, C) # Batch size, sequence length, number of embeddings (n_embed)
        
        # Apply the output linear layer
        return self.output_linear(contextual_embeddings)