import torch


@torch.no_grad()
def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy of the model.

    Parameters:
    - y_true (torch.Tensor): True target variable
    - y_pred (torch.Tensor): Predicted target variable

    Returns:
    - torch.Tensor: Accuracy torch.Tensor
    """

    # If the array is multi-dimensional, take the argmax
    if y_true.ndim > 1:
        y_true = torch.argmax(y_true, dim=-1)
    if y_pred.ndim > 1:
        y_pred = torch.argmax(y_pred, dim=-1)
    
    # Compute and return the accuracy as a torch.Tensor
    return torch.mean((y_true == y_pred).float(), dim=-1)