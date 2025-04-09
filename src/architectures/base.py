import torch
import torch.nn as nn
import numpy as np
from typing import Callable


class Architecture(nn.Module):
    
    ### Magic methods ###
    
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the architecture.
        """
        
        # Initialize the superclass
        super().__init__(*args, **kwargs)
        
        # Initialize the history of the model
        self.history = {}
        
        
    ### Public methods ###
    
    def init_history(self, metrics: list[Callable] = []) -> None:
        """
        Method to initialize the history of the model
        
        Parameters:
        - metrics (list[Callable]): List of metrics to evaluate the model
        """
        
        # Initialize the history of the model
        self.history = {
            "loss": torch.tensor(np.array([]), requires_grad=False, dtype=torch.float32),
            **{f"{metric.__name__}":  torch.tensor(np.array([]), requires_grad=False, dtype=torch.float32) for metric in metrics},
            "val_loss":  torch.tensor(np.array([]), requires_grad=False, dtype=torch.float32),
            **{f"val_{metric.__name__}":  torch.tensor(np.array([]), requires_grad=False, dtype=torch.float32) for metric in metrics}
        }
    
    
    def fit(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Method to fit the model.
        
        Returns:
        - dict[str, Tensor]: Dictionary containing the history of the model
        """
        
        # Raise an error if the method is not implemented
        raise NotImplementedError("The method 'fit' is not implemented. Please implement it in the child class.")