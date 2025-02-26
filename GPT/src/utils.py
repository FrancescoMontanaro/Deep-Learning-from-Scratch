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
    

# Set the device to use for the computations
device = get_device()