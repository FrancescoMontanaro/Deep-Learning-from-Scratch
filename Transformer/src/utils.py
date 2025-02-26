import os
import torch


def get_device() -> torch.device:
    """
    Function that returns the device to use for the computations.
    
    Returns:
    - device (torch.device): The device to use for the computations.
    """
    
    # Check if CUDA is available (for NVIDIA GPUs)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check if MPS is available (for Apple silicon chips)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    # No accelerators available, use the CPU
    else:
        return torch.device("cpu")
    

def load_txt_file(path: str) -> str:
    """
    Load a text file from the specified path.
    
    Parameters:
    - path (str): The path to the text file.
    
    Returns:
    - str: The contents of the text file.
    """
    
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f'The file "{path}" does not exist.')
    
    # Read the file
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()
    

# Set the device to use for the computations
device = get_device()