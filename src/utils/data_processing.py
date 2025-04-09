import torch
from typing import Union, Tuple


def shuffle_data(data: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Method to shuffle the dataset
    
    Parameters:
    - data (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Dataset to shuffle
    
    Returns:
    - Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Shuffled dataset
    
    Raises:
    - ValueError: If the data is not a torch.Tensor or a tuple
    """
    
    # Check if the data is a torch.Tensor or a tuple of torch.Tensors
    if isinstance(data, tuple) and len(data) == 2:
        # Unpack the data
        X, y = data
    
        # Get the number of samples
        n_samples = X.shape[0]
        
        # Generate random indices
        indices = torch.randperm(n_samples)
        
        # Shuffle the dataset
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    
        # Return the shuffled dataset
        return X_shuffled, y_shuffled
    
    # Check if the data is a torch.Tensor
    elif isinstance(data, torch.Tensor):
        # Get the number of samples
        n_samples = data.shape[0]
        
        # Generate random indices
        indices = torch.randperm(n_samples)
        
        # Shuffle the dataset
        data_shuffled = data[indices]
        
        # Return the shuffled dataset
        return data_shuffled
    
    else:
        # Raise a ValueError if the data is not a torch.Tensor or a tuple of torch.Tensors
        raise ValueError("data must be a torch.Tensor or a tuple of two torch.Tensors")