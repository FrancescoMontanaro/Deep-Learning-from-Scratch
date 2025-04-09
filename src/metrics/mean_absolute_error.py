import torch


@torch.no_grad()
def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean absolute error torch.torch.Tensor the model.

    Parameters:
    - y_true (torch.Tensor): True target variable
    - y_pred (torch.Tensor): Predicted target variable

    Returns:
    - torch.Tensor: Mean absolute error torch.torch.Tensor the model
    """
    
    # Compute and return the mean absolute error as a torch.Tensor
    return torch.mean(torch.abs(y_true - y_pred))