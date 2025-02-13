import torch

def calc_loss_batch(input_batch, target_batch, device, model):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    logits, target_batch = logits.flatten(0, 1), target_batch.flatten()  # flatten(start, end) Flattens dimensions from start to end idx specified
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_multi_batch(data_loader, device, model, num_batches = None):
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

    model.eval() # Set model to Evaluation mode
    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_multi_batch(train_loader, device, model, eval_batch_size)
        val_loss = calc_loss_multi_batch(validation_loader, device, model, eval_batch_size)
    model.train() # Reset model to Training mode

    return train_loss, val_loss
