import torch
from typing import Literal, Optional


class DataLoader:
    
    ### Magic methods ###
    
    def __init__(self, data: torch.Tensor, train_val_split: float, device: Optional[torch.device] = None) -> None:
        """
        Initialize the data loader.
        
        Parameters:
        - data (torch.Tensor): The data to load.
        - train_val_split (float): The proportion of the data to use for training.
        - device (torch.device): The device to use for the computations.
        """
        
        # Store the data
        self.data = data
        
        # Compute the split index
        n = int(train_val_split * len(data))

        # Split the data into training and validation sets
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        # Store the device
        self.device = device
    
    
    ### Public methods ###
    
    def get_batch(self, split: Literal["train", "validation"] = "validation", batch_size: int = 4, block_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method that returns a batch of data.
        
        Parameters:
        - split (str): The split to use for the data (either "train" or "validation").
        - batch_size (int): The size of the batch.
        - block_size (int): The size of the block (sequence length).
        
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: The input and target sequences.
        """
        
        # Select the data based on the split
        data = self.train_data if split == "train" else self.val_data
        
        # Randomly select the starting index for the sequence
        ix = torch.randint(len(data) - block_size, (batch_size,)) # (batch_size,)
        
        # Create the input and target sequences by selecting a block of text from the data
        # The target sequence is the input sequence shifted by one character
        x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (batch_size, block_size)
        
        # Check if the device is not None
        if self.device is not None:
            # Move the data to the appropriate device
            x, y = x.to(self.device), y.to(self.device)
        
        # Return the input and target sequences
        return x, y