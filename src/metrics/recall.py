import torch


@torch.no_grad()
def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the recall of the model.

    Parameters:
    - y_true (torch.Tensor): True target variable
    - y_pred (torch.Tensor): Predicted target variable

    Returns:
    - torch.Tensor: Recall of the model
    """
    
    # Compute the recall
    tp = torch.sum((y_true.data == 1) & (y_pred.data == 1))
    fn = torch.sum((y_true.data == 1) & (y_pred.data == 0))
    
    # Compute and return the recall as a torch.Tensor
    return tp / (tp + fn)