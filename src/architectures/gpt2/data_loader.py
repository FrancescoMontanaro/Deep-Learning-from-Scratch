import os
import torch
from typing import Literal, Optional

from .tokenizer import Tokenizer


class DataLoader:
    
    ### Magic methods ###
    
    def __init__(self, txt_file: str, tokenizer: Tokenizer, train_val_split: float = 0.1, device: Optional[torch.device] = None) -> None:
        """
        Initialize the data loader.
        
        Parameters:
        - txt_file (str): The path to the text file.
        - tokenizer (Tokenizer): The tokenizer to use.
        - train_val_split (float): The proportion of the data to use for training. Default is 0.1.
        - device (torch.device): The device to use for the tensors. Default is None.
        
        Raise:
        - FileNotFoundError: If the text file does not exist.
        """
        
        # Check if the text file exists
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"File not found: {txt_file}")
        
        # Read the text file
        with open(txt_file, "r") as f:
            text = f.read()
            
        # Instantiate the tokenizer
        self.tokenizer = tokenizer
        
        # Encode the text into tokens
        tokens = torch.tensor(tokenizer.encode(text))
        
        # Compute the number of training tokens
        n_training_tokens = int(train_val_split * len(tokens))
        
        # Split the tokens into training and validation sets
        self.train_tokens = tokens[:-n_training_tokens]
        self.val_tokens = tokens[-n_training_tokens:]
        
        # Store the device
        self.device = device
    
    
    ### Public methods ###
    
    def get_batch(self, split: Literal['train', 'valid'], batch_size: int = 4, sequence_length: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method that returns a batch of data.
        
        Parameters:
        - split (str): The split to use for the data (either "train" or "valid").
        - batch_size (int): The size of the batch.
        - block_size (int): The size of the block (sequence length).
        
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: The input and target sequences.
        """
        
        # Extract the tokens based on the split
        tokens = self.train_tokens if split == "train" else self.val_tokens
        
        # Randomly select the starting index for the sequence
        ix = torch.randint(len(tokens) - sequence_length, (batch_size,)) # (batch_size,)
        
        # Create the input and target sequences by selecting a block of text from the data
        # The target sequence is the input sequence shifted by one character
        x = torch.stack([tokens[i:i+sequence_length] for i in ix]) # (batch_size, block_size)
        y = torch.stack([tokens[i+1:i+sequence_length+1] for i in ix]) # (batch_size, block_size)
        
        # Check if the device is set
        if self.device is not None:
            # Move the data to the appropriate device
            x, y = x.to(self.device), y.to(self.device)
        
        # Return the input and target sequences
        return x, y