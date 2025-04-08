import torch
import torch.nn as nn
from torch.nn import functional as F


class SingleHeadAttention(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, head_size: int, n_embed: int, block_size: int, dropout: float = 0.1) -> None:
        """
        Initialize the single-head attention mechanism.
        
        Parameters:
        - head_size (int): The size of the attention head.
        - n_embed (int): The size of the embeddings.
        - block_size (int): The size of the block.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the superclass
        super().__init__()
        
        # Store the head size
        self.head_size = head_size
        
        # Define the key, query and value matrices
        self.key = nn.Linear(n_embed, head_size, bias=False) # (n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size, bias=False) # (n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size, bias=False) # (n_embed, head_size)
        
        # Creating a dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Registering a triangular mask as a buffer
        self.register_buffer("attention_mask", torch.tril(torch.ones(block_size, block_size))) # (block_size, block_size)
        
    
    ### Public methods ###
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the single-head attention mechanism.
        
        Parameters:
        - embeddings (torch.Tensor): The input embeddings.
        
        Returns:
        - torch.Tensor: The contextual embeddings.
        """
        
        # Extracting the shape of the embeddings
        B, T, C = embeddings.shape # B: batch size, T: sequence length, C: number of channels (n_embed)
        
        # Apply the key, query and value matrices to the embeddings
        k = self.key(embeddings) # (B, T, H)
        q = self.query(embeddings) # (B, T, H)
        v = self.value(embeddings) # (B, T, H)
        
        # Compute the attention weights, apply the attention mask and normalize the weights
        attention_weights = k @ q.transpose(-2, -1) * self.head_size ** -0.5 # (B, T, H) @ (B, H, T) = (B, T, T)
        attention_weights = attention_weights.masked_fill(self.attention_mask[:T, :T] == 0, float('-inf')) # (B, T, T) # type: ignore
        attention_weights = F.softmax(attention_weights, dim=-1) # (B, T, T)
        
        # Apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Compute the contextual embeddings by applying the attention weights to the value embeddings
        contextual_embeddings = attention_weights @ v # (B, T, T) @ (B, T, H) = (B, T, H)
        
        # Return the contextual embeddings
        return contextual_embeddings
    
    
class MultiHeadAttention(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, n_heads: int, head_size: int, n_embed: int, block_size: int, dropout : float = 0.1) -> None:
        """
        Initialize the multi-head attention mechanism.
        
        Parameters:
        - n_heads (int): The number of attention heads.
        - head_size (int): The size of each attention head.
        - n_embed (int): The size of the embeddings.
        - block_size (int): The size of the block.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the superclass
        super().__init__()
        
        # Creating the heads
        self.heads = nn.ModuleList([SingleHeadAttention(head_size, n_embed, block_size) for _ in range(n_heads)])
        
        # Create the output linear layer
        self.output_linear = nn.Linear(n_embed, n_embed) # Project the embeddings back to the original size
        
        # Create the dropout layer
        self.dropout = nn.Dropout(dropout)
        

    ### Public methods ###
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head attention mechanism.
        
        Parameters:
        - embeddings (torch.Tensor): The input embeddings.
        
        Returns:
        - torch.Tensor: The output embeddings.
        """
        
        # Apply each head to the embeddings
        out = torch.cat([head(embeddings) for head in self.heads], dim=-1) # (B, T, H * n_heads)
        
        # Apply the output linear layer
        out = self.dropout(self.output_linear(out)) # (B, T, C)
        
        return out