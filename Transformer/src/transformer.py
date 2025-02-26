import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

from .utils import device
from .data_loader import DataLoader
from .feed_forward import FeedForward
from .attention_mechanism import MultiHeadAttention


class TransformerBlock(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, n_embed: int, n_heads: int, block_size: int, dropout: float = 0.1) -> None:
        """
        Initialize the transformer block.
        
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
        self.feed_forward = FeedForward(  # Create the feed-forward layers
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


class Transformer(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, vocab_size: int, n_embed: int, n_heads: int, block_size: int, n_transformer_blocks: int = 4, dropout: float = 0.1) -> None:
        """
        Initialize the Transformer.
        
        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - n_embed (int): The size of the embeddings (dimensionality).
        - n_heads (int): The number of attention heads.
        - block_size (int): The size of the block.
        - n_transformer_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        """
        
        # Initialize the superclass
        super().__init__()
        
        # Store the block size
        self.block_size = block_size
        
        ### DECODER ###
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # Create the token embedding table
        self.positional_embedding = nn.Embedding(block_size, n_embed) # Create the positional embedding table
        self.transformer_blocks = nn.Sequential(*[  # Create the transformer blocks
            TransformerBlock(
                n_embed = n_embed, 
                n_heads = n_heads, 
                block_size = block_size,
                dropout = dropout
            ) for _ in range(n_transformer_blocks)
        ], nn.LayerNorm(n_embed))  # Add a normalization at the end of the encoder
        self.head = nn.Linear(n_embed, vocab_size) # Create the linear layer to get the logits
        
        ###############
        
        # Move the model to the appropriate device
        self.to(device)
        
        # Print the device
        print(f'Model moved to device: {device}')
        
    
    ### Public methods ###
    
    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Parameters:
        - input_tokens (torch.Tensor): The input tokens.
        - targets (torch.Tensor): The target tokens.
        
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: The logits and the loss.
        """
        
        # Extracting the shape of the input tokens
        B, T = input_tokens.shape
        
        # Get the token embeddings
        token_embeddings = self.token_embedding_table(input_tokens) # (B, T, n_embed)
        
        # Get the positional embeddings
        positions = self.positional_embedding(torch.arange(T, device=input_tokens.device)) # (T, n_embed)
        
        # Additively combine the token and positional embeddings
        embeddings = token_embeddings + positions # (B, T, n_embed)
        
        # Apply the self-attention mechanism
        embeddings = self.transformer_blocks(embeddings) # (B, T, n_embed)
        
        # Compute the logits
        logits = self.head(embeddings) # (B, T, vocab_size)
        
        if targets is None:
            # If no targets are provided, return the logits and no loss
            loss = None
            
        else:
            # Extracting the shape of the logits
            B, T, V = logits.shape
            
            # Reshaping the logits and targets to have the same shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            
            # Compute the loss
            loss = F.cross_entropy(logits, targets)
        
        # Return the logits
        return logits, loss
    
    
    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Method to generate new tokens (inference).
        
        Parameters:
        - input_tokens (torch.Tensor): The input tokens.
        - max_new_tokens (int): The maximum number of new tokens to generate.
        
        Returns:
        - torch.Tensor: The generated tokens.
        """
        
        # Iterate over the maximum number of new tokens
        for _ in range(max_new_tokens):
            # Crop the input tokens to the block size
            cropped_input_tokens = input_tokens[:, -self.block_size:] # (B, T)
            
            # Get the predictions
            logits, loss = self(cropped_input_tokens)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]
            
            # Apply the softmax function to get the probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token from the distribution
            input_tokens_next = torch.multinomial(probs, num_samples=1)
            
            # Concatenate the new token to the input
            input_tokens = torch.cat([input_tokens, input_tokens_next], dim=-1)
            
        return input_tokens
    
    
    def train_model(self, data_loader: DataLoader, epochs: int, lr: float, batch_size: int, eval_iters: int = 200) -> None:
        """
        Method to train the model.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - epochs (int): The number of epochs.
        - lr (float): The learning rate.
        - batch_size (int): The batch size.
        - eval_iters (int): The number of iterations to evaluate the loss on the training and validation sets.
        """
        
        # Define an optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # Iterate over the epochs
        for epoch in range(epochs):
            # Evaluate the loss on the training and validation sets every once in a while
            if epoch % eval_iters == 0:
                # Estimate the losses
                losses = self.estimate_loss(data_loader, eval_iters, batch_size)
                
                # Print the losses
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {losses["train"]:.4f}, Val Loss: {losses["val"]:.4f}')
                
            # Get the batch
            x, y = data_loader.get_batch(split='train', batch_size=batch_size, block_size=self.block_size)
            
            # Get the logits and loss
            _, loss = self(x, y)
            
            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Perform the backward pass
            loss.backward()
            
            # Update the parameters
            optimizer.step()
        

    @torch.no_grad()
    def estimate_loss(self, data_loader: DataLoader, eval_iters: int, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Method to estimate the loss on the training and validation sets.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - eval_iters (int): The number of iterations to evaluate the loss.
        - batch_size (int): The batch size.
        
        Returns:
        - dict[str, torch.Tensor]: The estimated losses.
        """
        
        # Initialize the output dictionary
        out = {}
        
        # Set the model to evaluation mode
        self.eval()
        
        # Iterate over the splits
        for split in ['train', 'val']:
            # Initialize the losses tensor
            losses = torch.zeros(eval_iters)
            
            # Iterate over the evaluation iterations
            for k in range(eval_iters):
                # Getting a batch of data
                x, y = data_loader.get_batch(split=split, batch_size=batch_size, block_size=self.block_size) # type: ignore
                
                # Get the logits and loss
                logits, loss = self(x, y)
                
                # Store the loss
                losses[k] = loss.item()
            
            # Compute the mean loss
            out[split] = losses.mean()
            
        # Set the model back to training mode
        self.train()
        
        return out