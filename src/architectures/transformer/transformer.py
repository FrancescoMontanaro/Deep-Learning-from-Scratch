import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Union, Generator

from .data_loader import DataLoader
from .decoder_block import DecoderBlock


class Transformer(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, vocab_size: int, n_embed: int, n_heads: int, block_size: int, n_transformer_blocks: int = 4, dropout: float = 0.1, device: Optional[torch.device] = None) -> None:
        """
        Initialize the Transformer.
        
        Parameters:
        - vocab_size (int): The size of the vocabulary.
        - n_embed (int): The size of the embeddings (dimensionality).
        - n_heads (int): The number of attention heads.
        - block_size (int): The size of the block.
        - n_transformer_blocks (int): The number of transformer blocks.
        - dropout (float): The dropout rate.
        - device (torch.device): The device to use for the computations.
        """
        
        # Initialize the superclass
        super().__init__()
        
        # Store the block size
        self.block_size = block_size
        
        ### DECODER ###
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # Create the token embedding table
        self.positional_embedding = nn.Embedding(block_size, n_embed) # Create the positional embedding table
        self.transformer_blocks = nn.Sequential(*[  # Create the transformer blocks
            DecoderBlock(
                n_embed = n_embed, 
                n_heads = n_heads, 
                block_size = block_size,
                dropout = dropout
            ) for _ in range(n_transformer_blocks)
        ], nn.LayerNorm(n_embed))  # Add a normalization at the end of the encoder
        self.head = nn.Linear(n_embed, vocab_size) # Create the linear layer to get the logits
        
        ###############
        
        # Create the loss function
        self.loss_fn = nn.CrossEntropyLoss() 
        
        ## Move the model to the appropriate device
        if device is not None:
            # Move the model to the appropriate device
            self.to(device)
        
    
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
            loss = self.loss_fn(logits, targets)
        
        # Return the logits
        return logits, loss
        
    
    def fit(self, data_loader: DataLoader, steps: int, lr: float, batch_size: int, eval_iters: int = 200) -> None:
        """
        Method to train the model.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - steps (int): The number of steps.
        - lr (float): The learning rate.
        - batch_size (int): The batch size.
        - eval_iters (int): The number of iterations to evaluate the loss on the training and validation sets.
        """
        
        # Define an optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        
        # Iterate over the steps
        for step in range(steps):
            # Evaluate the loss on the training and validation sets every once in a while
            if step % eval_iters == 0:
                # Estimate the losses
                losses = self.estimate_loss(data_loader, eval_iters, batch_size)
                
                # Print the losses
                print(f'Epoch {step+1}/{steps} - Train Loss: {losses["train"]:.4f}, Val Loss: {losses["val"]:.4f}')
                
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
    
    
    @torch.no_grad()
    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int, stream: bool = False) -> Union[torch.Tensor, Generator[torch.Tensor, None, None]]:
        """
        Method to generate new tokens (inference).
        
        Parameters:
        - input_tokens (Tensor): The input tokens.
        - max_new_tokens (int): The maximum number of new tokens to generate.
        - stream (bool): Whether to generate the tokens in a streaming fashion.
        
        Returns:
        - Union[Tensor, Generator[Tensor, None, None]]: The generated tokens or a generator to stream the tokens.
        """
        
        # Set the model to evaluation mode
        self.eval()
            
        # Stream the generated tokens using a generator
        if stream:
            # Define the generator to stream the tokens
            def stream_tokens() -> Generator[torch.Tensor, None, None]:
                """
                Generator to stream the generated tokens.
                
                Yields:
                - Tensor: The next token generated.
                """
                
                # Initialize the local tokens
                local_tokens = input_tokens
                
                # Iterate over the maximum number of new tokens
                for _ in range(max_new_tokens):
                    # Crop the input tokens to the sequence length if larger
                    cropped_input_tokens = local_tokens[:, -self.block_size:]
                    
                    # Get the predictions
                    logits, _ = self(cropped_input_tokens)
                    
                    # Focus only on the last time step
                    logits = logits[:, -1, :]
                    
                    # Apply the softmax function to get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    
                    # Sample the next token from the distribution
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Yield the next token
                    yield next_token

                    # Concatenate the new token to the input
                    local_tokens = torch.cat([local_tokens, next_token], dim=-1)
            
            # Return the generator to stream the tokens         
            return stream_tokens()
        
        # Generate all the tokens at once
        else:
            # Iterate over the maximum number of new tokens
            for _ in range(max_new_tokens):
                # Crop the input tokens to the sequence length if larger
                cropped_input_tokens = input_tokens[:, -self.block_size:]
                
                # Get the predictions
                logits, _ = self(cropped_input_tokens)
                
                # Focus only on the last time step
                logits = logits[:, -1, :]
                
                # Apply the softmax function to get the probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample the next token from the distribution
                next_token = torch.multinomial(probs, num_samples=1)

                # Concatenate the new token to the input
                input_tokens = torch.cat([input_tokens, next_token], dim=-1)
                
            # Return the generated tokens
            return input_tokens