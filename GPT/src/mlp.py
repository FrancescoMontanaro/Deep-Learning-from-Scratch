import torch
import torch.nn as nn

from .config import GPTConfig


class MultiLayerPerceptron(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, config: GPTConfig) -> None:
        """
        Class constructor
        
        Parameters:
        - config (GPTConfig): The configuration class for the GPT model.
        """
        
        # Initialize the parent class
        super().__init__()
        
        # Creating the layers of the multi-layer perceptron
        self.linear_1 = nn.Linear(config.n_embed, 4 * config.n_embed) # Project the embeddings to a higher-dimensional space
        self.activation = nn.GELU(approximate='tanh') # Apply the GELU activation function
        self.project = nn.Linear(4 * config.n_embed, config.n_embed) # Project the embeddings back to the original size
        
        # Add a flag to the output linear layer to indicate that it is a residual layer
        self.project.residual_layer = 1 # type: ignore
     

    ### Public methods ###   
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer perceptron.
        
        Parameters:
        - x (torch.Tensor): The input embeddings.
        
        Returns:
        - torch.Tensor: The output embeddings.
        """
        
        # Apply the feed-forward layers
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.project(x)
        
        # Return the output embeddings
        return x