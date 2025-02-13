from M2_pretraining.evaluate import calc_loss_batch, evaluate_model
from M2_pretraining.helpers import generate_print_sample 

def train_model(model, device,  #  Model Related inputs
                num_epochs, optimizer,  #  Parameters needed for training
                train_data_loader, validation_data_loader, tokenizer,  # Train & Validation data related
                test_string, eval_freq, eval_batch_size):  # Evlauation required inputs
    # ======== These variables are Optional - To track losses ========
    global_step = -1  # Starting from -1 so that first step is printed
    tokens_seen = 0

    train_losses, validation_losses, tokens_seen_tracker = [], [], []  # Collected to print plot later - To be returned
    # ================================================================

    for epoch in range(num_epochs):
        model.train()  # Set model to train mode

        for input_batch, target_batch in train_data_loader:
            optimizer.zero_grad()  # 1. Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, device, model)  # 2. Calculate loss of the batch
            loss.backward()  # 3. Calculate loss gradients
            optimizer.step() # 4. Update model weights using loss gradients

            # ======== Below section is optional to track losses ========
            global_step += 1
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, validation_loss = evaluate_model(model, train_data_loader, validation_data_loader, device, eval_batch_size)
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)
                tokens_seen_tracker.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {validation_loss:.3f}")

            # ========================================================

        # Print sample after each epoch to observe improvement in train quality
        print(f"Epoch: {epoch+1}")
        generate_print_sample(model, test_string, tokenizer)

    return train_losses, validation_losses, tokens_seen_tracker
