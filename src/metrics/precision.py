import torch


@torch.no_grad()
def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the precision of the model.

    Parameters:
    - y_true (torch.Tensor): True target variable
    - y_pred (torch.Tensor): Predicted target variable

    Returns:
    - torch.Tensor: Precision of the model
    """
    
    # Compute the precision
    tp = torch.sum((y_true == 1) & (y_pred == 1))
    fp = torch.sum((y_true == 0) & (y_pred == 1))
    
    # Compute and return the precision as a torch.Tensor
    return tp / (tp + fp)