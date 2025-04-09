import torch

from .recall import recall
from .precision import precision


@torch.no_grad()
def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the F1 score of the model.

    Parameters:
    - y_true (torch.Tensor): True target variable
    - y_pred (torch.Tensor): Predicted target variable

    Returns:
    - float: F1 score of the model
    """
    
    # Compute the precision and recall
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    # Compute the F1 score
    f1 = 2 * (prec * rec) / (prec + rec)

    # Compute and return the F1 score as a torch.Tensor
    return f1