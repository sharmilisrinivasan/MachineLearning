"""
This file contains the model evaluation methods used during training.
It uses the cross-entropy loss function and computes both training and validation loss.
"""
import torch

def calc_loss_batch(input_batch, target_batch, device, model):
    """
    Calculate the loss for a single batch of data.
    Args:
        input_batch: The input data for the batch.
        target_batch: The target data for the batch.
        device: The device to run the model on (CPU or GPU).
        model: The model to evaluate.
    Returns:
        The calculated loss for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    logits, target_batch = logits.flatten(0, 1), target_batch.flatten()  # flatten(start, end) Flattens dimensions from start to end idx specified
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_multi_batch(data_loader, device, model, num_batches = None):
    """
    Calculate the loss for multiple batches of data.
    Args:
        data_loader: The data loader for the batches, containing input and target batches.
        device: The device to run the model on (CPU or GPU).
        model: The model to evaluate.
        num_batches: The number of batches to evaluate. If None, evaluates all batches.
    Returns:
        The average loss for the evaluated batches.
    """
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # If no num_btaches given, then consider complete batch
        num_batches = len(data_loader)
    else:
        # if num_batches > batch_size, we can take complete batch
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.
    for index, (input_batch, target_batch) in enumerate(data_loader):
        if index >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, device, model)
    return total_loss / num_batches

def evaluate_model(model, train_loader, validation_loader, device, eval_batch_size = None):
    """
    Evaluate the model on the training and validation data.
    Args:
        model: The model to evaluate.
        train_loader: The data loader for the training data.
        validation_loader: The data loader for the validation data.
        device: The device to run the model on (CPU or GPU).
        eval_batch_size: The number of batches to evaluate. If None, evaluates all batches.
    Returns:
        The average training loss and validation loss.
    """

    model.eval() # Set model to Evaluation mode
    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_multi_batch(train_loader, device, model, eval_batch_size)
        val_loss = calc_loss_multi_batch(validation_loader, device, model, eval_batch_size)
    model.train() # Reset model to Training mode

    return train_loss, val_loss
