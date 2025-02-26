import time
import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

from .utils import *
from .block import Block
from .config import GPTConfig
from .data_loader import DataLoader


class GPT2(nn.Module):
    
    ### Magic Methods ###
    
    def __init__(self, config: GPTConfig) -> None:
        """
        Class constructor for the GPT2 model.
        
        Parameters:
        - config (GPTConfig): The configuration class for the GPT model.
        """
        
        # Initialize the parent class
        super().__init__()
        
        # Store the configuration
        self.config = config
        
        # Create the model layers
        self.transformer = nn.ModuleDict(dict(
            tokens_embedding = nn.Embedding(config.vocab_size, config.n_embed),
            positional_embedding = nn.Embedding(config.context_size, config.n_embed),
            hidden = nn.ModuleList([Block(config) for _ in range(config.n_blocks)]),
            normalization = nn.LayerNorm(config.n_embed)
        ))
        
        # Create the output layer to project the embeddings back to the vocabulary size
        self.output = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # Weights sharing between the tokens embedding layer and the output layer
        # This is done to reduce the number of parameters in the model, improve the performance and
        # have a semantic relationship between the input and output tokens.
        self.transformer.tokens_embedding.weight = self.output.weight # type: ignore
        
        # Initialize the parameters of the model
        self.apply(self._inti_params)
        
        
    ### Public methods ###  
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT2 model.
        
        Parameters:
        - x (torch.Tensor): The input tokens.
        - y (torch.Tensor, optional): The target tokens. Default is None.
        
        Returns:
        - tuple[torch.Tensor, Optional[torch.Tensor]]: The logits and the loss value.
        """
        
        # Extract the shape of the input tensor
        B, T = x.size()
        
        # Check if the input sequence length is greater than the context size
        if T > self.config.context_size:
            # Take only the last context size tokens
            x = x[:, -self.config.context_size:]
        
        # Create the positional encodings
        position = torch.arange(T, device=x.device).expand(B, T)
        
        # Apply the token and positional embeddings to the input tokens
        x = self.transformer.tokens_embedding(x) + self.transformer.positional_embedding(position) # type: ignore
        
        # Apply the transformer blocks
        for block in self.transformer.hidden: # type: ignore
            # Apply the hidden block
            x = block(x)
            
        # Apply the normalization layer
        x = self.transformer.normalization(x) # type: ignore
        
        # Apply the output layer to project the embeddings back to the vocabulary size
        logits = self.output(x) # logits (B, T, vocab_size)
        
        # Initialize the loss value
        loss = None
        
        if y is not None:
            # Calculate the loss value
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) # (B*T, vocab_size) -> (B*T)
        
        # Return the logits and the loss value
        return logits, loss
    
    
    def fit(self, data_loader: DataLoader, epochs: int, lr: float, batch_size: int = 4, sequence_length: int = 8, micro_batch_size: Optional[int] = None) -> None:
        """
        Method to train the model.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - epochs (int): The number of epochs.
        - lr (float): The learning rate.
        - batch_size (int): The batch size.
        - sequence_length (int): The sequence length.
        - micro_batch_size (int, optional): The size of the micro-batch. Default is None.
        
        Raises:
        - AssertionError: If the batch size is not divisible by the micro-batch size * sequence length.
        """
        
        # Check if the micro-batch size is provided
        if micro_batch_size is None:
            # Set the micro-batch size to the batch size if not provided
            micro_batch_size = batch_size
        else:
            # Checking that the batch size is divisible by the micro-batch size
            assert batch_size % (micro_batch_size * sequence_length) == 0, "The batch size must be divisible by the micro-batch size * sequence length. (mB * T)"
        
        # Define an optimizer
        optimizer = torch.optim.AdamW(
            params = self.parameters(), 
            lr = lr, 
            betas = (0.9, 0.95), # Default values from the GPT-3 paper,
            eps = 1e-8 # Default value from the GPT-3 paper
        )
        
        # Compute the number of steps per epoch and micro-steps per batch
        n_steps = max(1, len(data_loader.train_tokens) // (batch_size * sequence_length))
        n_micro_steps = max(1, batch_size // (micro_batch_size * sequence_length))
        
        # Iterate over the epochs
        for epoch in range(epochs):
            # Initialize the epoch loss and the step duration
            epoch_loss = 0.0
            step_duration = 0.0
            
            # Start the timer of the epoch
            epoch_start = time.time()
            
            # Iterate over the steps
            for step in range(n_steps):
                # Start the timer
                t0 = time.time()
                
                # Reset the gradients
                optimizer.zero_grad()
                
                # Initialize the loss accumulator
                loss_accum = 0.0
                
                # Iterate over the micro-steps
                for micro_step in range(n_micro_steps):
                    # Get the current batch
                    x, y = data_loader.get_batch('train', micro_batch_size, sequence_length)
            
                    # Perform the forward pass
                    _, loss = self(x, y)
                    
                    # Normalize the loss value by the micro-batch size
                    # This is done because the gradient accumulation only sums up the gradients without normalizing them
                    loss /= n_micro_steps
                    
                    # Accumulate the loss
                    loss_accum += loss.detach()
                
                    # Compute the gradients
                    loss.backward()
                    
                    # Display the step progress
                    print(f"\rEpoch {epoch + 1}/{epochs} | Step {step + 1}/{n_steps} | Step completion percentage {(((micro_step + 1) / n_micro_steps)*100):.2f}%", end="")
                
                # Clip the gradients to avoid exploding gradients
                # The gradient norm is clipped to 1.0
                # This technique is taken from the GPT-3 paper
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                
                # Update the parameters
                optimizer.step()
                
                # Accumulate epoch loss
                epoch_loss += loss_accum
                
                # Accumulate the step duration
                t1 = time.time() # Stop the timer
                dt = (t1 - t0) * 1000 # Compute the elapsed time in milliseconds
                step_duration += dt # Accumulate the step duration
                
                # Display the epoch progress
                print(f"\rEpoch: {epoch + 1}/{epochs} | Completion percentage: {(((step + 1)/n_steps)*100):.2f}% | Step duration {dt:.2f} ms/step --> loss: {loss_accum:.4f}")
            
            # Compute the average epoch loss and step duration  
            avg_epoch_loss = epoch_loss / n_steps
            avg_step_duration = step_duration / n_steps
            
            # Evaluate the model on the validation set
            val_loss = self.estimate_loss(data_loader, micro_batch_size, sequence_length)
            
            # Stop the timer of the epoch
            epoch_end = time.time()
            
            # Compute the epoch duration
            epoch_duration = (epoch_end - epoch_start) * 1000
              
            # Display the epoch loss  
            print(f"Epoch {epoch + 1}/{epochs} | Average step duration {avg_step_duration:.2f} ms/step | Epoch duration {epoch_duration:.2f} ms/epoch --> loss: {avg_epoch_loss:.4f} - val_loss: {val_loss:.4f}")
        
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Method to generate text from the GPT2 model.
        
        Parameters:
        - x (torch.Tensor): The input tokens.
        - max_new_tokens (int): The maximum number of tokens to generate.
        
        Returns:
        - torch.Tensor: The generated tokens.
        """
        
        # Loop until the maximum number of tokens is generated
        while x.size(1) < max_new_tokens:
            # Generate the next token
            logits, _ = self(x) # (B, vocab_size)
            
            # Get the last token logits
            logits = logits[:, -1] # (B, vocab_size)
            
            # Apply the softmax function to the logits
            probs = torch.softmax(logits, dim=-1)
            
            # Do the top-k sampling of 50 tokens
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            # Select a random token from the k tokens with the highest probabilities.
            # By using the multinomial function from the top k tokens with the highest probabilities,
            # we can select the next token by adding a bit of randomness to the selection process,
            # but still bias the selection towards the tokens with the highest probabilities.
            next_token = torch.multinomial(topk_probs, 1) # (B, 1)
            
            # Get the index of the next token
            next_token = torch.gather(topk_indices, -1, next_token) # (B, 1)
            
            # Concatenate the next token to the input tokens
            x = torch.cat((x, next_token), dim=-1)
            
        # Return the generated tokens
        return x
    
    
    @torch.no_grad()
    def estimate_loss(self, data_loader: DataLoader, batch_size: int, sequence_length: int) -> float:
        """
        Method to estimate the loss on the training and validation sets.
        
        Parameters:
        - data_loader (DataLoader): The data loader object to get the data from.
        - batch_size (int): The batch size.
        - sequence_length (int): The sequence length.
        
        Returns:
        - float: The loss value on the validation set.
        """
        
        # Set the model to evaluation mode
        self.eval()
        
        # Compute the number of evaluation steps
        n_steps = max(1, len(data_loader.val_tokens) // (batch_size * sequence_length))
        
        # Initialize the loss values
        val_loss = 0.0
        
        # Iterate over the validation steps
        for _ in range(n_steps):
            # Get the current batch
            x, y = data_loader.get_batch('valid', batch_size, sequence_length)
            
            # Perform the forward pass
            _, loss = self(x, y)
            
            # Accumulate the loss
            val_loss += loss.item()
            
        # Compute the average validation loss
        avg_val_loss = val_loss / n_steps
        
        # Set the model back to training mode
        self.train()
        
        # Return the validation loss
        return avg_val_loss
        

    ### Protected methods ###
    
    def _inti_params(self, module: nn.Module) -> None:
        """
        Method to initialize the parameters of the model.
        
        Parameters:
        - module (nn.Module): The module to initialize the parameters.
        """
        
        # Linear layer
        if isinstance(module, nn.Linear):
            # Default standard deviation for the weights
            std = 0.02
            
            # Check if the module is a residual layer, according to the GPT2 paper
            if hasattr(module, 'residual_layer'):
                # Increase the standard deviation for the weights
                std *= (2 * self.config.n_blocks) ** -0.5
            
            # Initialize the weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
            # Initialize the bias, if it exists
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # Embedding layer           
        elif isinstance(module, nn.Embedding):
            # Initialize the weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)