import torch.nn as nn
from typing import Any


class Callback:
    
    ### Magic methods ###
    
    def __call__(self, module: nn.Module) -> Any:
        """
        This method is called when the callback is called.
        
        Parameters:
        - module (Module): The module that is being trained.
        
        Returns:
        - Any: the output of the callback.
        """
        
        pass