import torch


@torch.no_grad()
def confusion_matrix(num_classes: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the confusion matrix of the model.

    Parameters:
    - num_classes (int): Number of classes
    - y_true (torch.Tensor): True target variable
    - y_pred (torch.Tensor): Predicted target variable

    Returns:
    - torch.Tensor: Confusion matrix
    """
    
    # Arrays must be 1D
    if y_true.ndim > 1:
        y_true = torch.argmax(y_true, dim=-1)
    if y_pred.ndim > 1:
        y_pred = torch.argmax(y_pred, dim=-1)
    
    # Compute the confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes))
    
    # Fill the confusion matrix
    for i in range(len(y_true)):
        # Increment the confusion matrix
        confusion_matrix[y_true[i], y_pred[i]] += 1
        
    # Return the confusion matrix as a torch.Tensor
    return confusion_matrix