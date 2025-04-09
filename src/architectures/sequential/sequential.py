import math
import time
import torch
import torch.nn as nn
from typing import Callable

from ..base import Architecture
from ...utils.data_processing import shuffle_data


class Sequential(Architecture, nn.Sequential):
    
    ### Magic methods ###
    
    def __init__(self, *args):
        """
        Initialize the Sequential model.
        """
        
        # Initialize the parent class class
        super(Sequential, self).__init__(*args)
        
        
    def fit(
        self, 
        X_train: torch.Tensor, 
        y_train: torch.Tensor,
        X_valid: torch.Tensor,
        y_valid: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 10,
        metrics: list[Callable] = [],
        callbacks: list[Callable] = []
    ) -> dict[str, torch.Tensor]:
        """
        Method to train the neural network
        
        Parameters:
        - X_train (Tensor): Features of the training dataset. Shape: (samples, ...)
        - y_train (Tensor): Labels of the training dataset. Shape: (samples, ...)
        - X_valid (Tensor): Features of the validation dataset. Shape: (samples, ...)
        - y_valid (Tensor): Labels of the validation dataset. Shape: (samples, ...)
        - optimizer (Optimizer): Optimizer to update the parameters of the model
        - loss_fn (LossFn): Loss function to compute the error of the model
        - batch_size (int): Number of samples to use for each batch. Default is 8
        - gradient_accumulation_steps (int): Number of steps to accumulate the gradients before updating the parameters. Default is 1
        - epochs (int): Number of epochs to train the model. Default is 10
        - metrics (list[Callable]): List of metrics to evaluate the model. Default is an empty list
        - callbacks (list[Callback]): List of callbacks to execute
        
        Returns:
        - dict[str, Tensor]: Dictionary containing the training and validation losses
        """
        
        #######################
        ### Initializations ###
        #######################
        
        # Initialize the history of the model
        self.init_history(metrics)
        
        # Initialize the control variables
        self.epoch, self.stop_training = 0, False
        n_training_steps = max(1, math.ceil(X_train.shape[0] / batch_size))
        n_valid_steps = max(1, math.ceil(X_valid.shape[0] / batch_size))
            
        ################################
        ### Start main training loop ###
        ################################
        
        # Iterate over the epochs
        while self.epoch < epochs and not self.stop_training:
            
            ############################
            ### Start training phase ###
            ############################
            
            # Set the model in training mode
            self.train()
            
            # Shuffle the dataset at the beginning of each epoch
            X_train_shuffled, Y_train_shuffled = shuffle_data((X_train, y_train))
            
            # Iterate over the batches
            elapsed_time = 0.0
            training_epoch_loss = torch.tensor(0.0, requires_grad=False)
            train_metrics = {metric.__name__: 0.0 for metric in metrics}
            for training_step in range(n_training_steps):
                # Store the start time
                start_time = time.time()
                
                # Get the current batch of data
                X_training_batch = X_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                y_training_batch = Y_train_shuffled[training_step * batch_size:(training_step + 1) * batch_size]
                
                # Forward pass: Compute the output of the model
                training_batch_output = self.forward(X_training_batch)
                
                # Compute the loss of the model
                training_loss = loss_fn(training_batch_output, y_training_batch)
                
                # Check if the number of accumulation steps is greater than 1
                if gradient_accumulation_steps > 1:
                    # Scale the loss by the number of accumulation steps
                    training_loss /= gradient_accumulation_steps
                    
                # Execute the backward pass
                training_loss.backward()
                
                # If the number of accumulation steps is reached or it is the last step, update the parameters
                if training_step % gradient_accumulation_steps == 0 or training_step == n_training_steps - 1:
                    # Update the parameters of the model
                    optimizer.step()
                    
                    # Zero the gradients of the parameters
                    optimizer.zero_grad()
                
                # Update the epoch loss
                training_epoch_loss += training_loss.detach().item()
                
                # Compute the metrics
                for metric in metrics:
                    train_metrics[metric.__name__] += metric(y_training_batch, training_batch_output)
                    
                # Comute the the statistics
                end_time = time.time() # Store the end time
                elapsed_time += (end_time - start_time) # Update the elapsed time
                ms_per_step = elapsed_time / (training_step + 1) * 1000 # Compute the milliseconds per step
                
                # Display epoch progress
                print(f"\rEpoch {self.epoch + 1}/{epochs} ({round((((training_step + 1)/n_training_steps)*100), 2)}%) | {round(ms_per_step, 2)} ms/step --> loss: {training_loss.item():.4f}", end="")
            
            ##############################
            ### Start validation phase ###
            ##############################
            
            # Set the model in evaluation mode
            self.eval()
            
            # Disable automatic gradient computation
            with torch.no_grad(): 
                # Iterate over the validation steps
                valid_epoch_loss = 0.0
                valid_metrics = {metric.__name__: 0.0 for metric in metrics}
                for valid_step in range(n_valid_steps):
                    # Get the current batch of validation data
                    X_valid_batch = X_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                    y_valid_batch = y_valid[valid_step * batch_size:(valid_step + 1) * batch_size]
                
                    # Compute the output of the model for the current validation batch
                    valid_batch_output = self.forward(X_valid_batch)
                    
                    # Compute the loss of the model for the current validation batch
                    # and update the validation epoch loss
                    valid_epoch_loss += loss_fn(valid_batch_output, y_valid_batch)
                    
                    # Compute the metrics
                    for metric in metrics:
                        valid_metrics[metric.__name__] += metric(y_valid_batch, valid_batch_output)
                
                ##########################
                ### Store the results  ###
                ##########################
                    
                # Store the training and validation losses
                self.history["loss"] = torch.cat((self.history["loss"], torch.tensor([training_epoch_loss / n_training_steps])), dim=0)
                self.history["val_loss"] = torch.cat((self.history["val_loss"], torch.tensor([valid_epoch_loss / n_valid_steps])), dim=0)
                
                # Compute the average metrics
                for metric in metrics:
                    # Compute the average of the metrics for the training and validation sets and store them
                    self.history[metric.__name__] = torch.cat((self.history[metric.__name__], torch.tensor([train_metrics[metric.__name__] / n_training_steps])))
                    self.history[f"val_{metric.__name__}"] = torch.cat((self.history[f"val_{metric.__name__}"], torch.tensor([valid_metrics[metric.__name__] / n_valid_steps])))
            
            #############################
            ### Display the progress  ###
            #############################
            
            # Display progress with metrics
            print(
                f"\rEpoch {self.epoch + 1}/{epochs} --> "
                f"loss: {self.history['loss'][-1].item():.4f} "
                + " ".join(
                    [f"- {metric.__name__.replace('_', ' ')}: {self.history[metric.__name__][-1].item():.4f}" for metric in metrics]
                )
                + f" | Validation loss: {self.history['val_loss'][-1].item():.4f} "
                + " ".join(
                    [f"- Validation {metric.__name__.replace('_', ' ')}: {self.history[f'val_{metric.__name__}'][-1].item():.4f}" for metric in metrics]
                ).ljust(50)
            )
            
            #############################
            ### Execute the callbacks ###
            #############################
            
            # Increment the epoch counter
            self.epoch += 1
                    
            # Execute the callbacks
            for callback in callbacks:
                # Call the callback
                callback(self)
         
        # Return the history of the training   
        return self.history