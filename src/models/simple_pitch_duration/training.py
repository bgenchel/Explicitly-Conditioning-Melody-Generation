import torch

def train_epoch(model, data_iter, data_assembler, loss_fn, optimizer, print_interval=250):
    batch_count = 0
    interval_loss = 0.0
    epoch_loss = 0.0
    for data_batch, target_batch in data_iter:
        # get the data in the right form
        data_batch, target_batch = map(data_assembler, (data_batch, target_batch))
        # detach hidden state
        model.repackage_hidden_and_cell()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        output = model(data_batch)[:, -1, :]
        # backward + optimize
        loss = loss_fn(output, target_batch)
        loss.backward()
        optimizer.step()
        # print stats out
        epoch_loss += float(loss.item())
        interval_loss += float(loss.item())
        if batch_count % print_interval == print_interval - 1:
            print("Batch: %d, Loss: %.5f" % (batch_count + 1, interval_loss / print_interval))
            interval_loss = 0.0
        batch_count += 1
    avg_epoch_loss = epoch_loss / batch_count
    return avg_epoch_loss

def eval_epoch(model, data_iter, data_assembler, loss_fn):
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for data_batch, target_batch in data_iter:
            data_batch, target_batch = map(data_assembler, (data_batch, target_batch))
            output = model(data_batch)[:, -1, :]
            loss = loss_fn(output, target_batch)
            total_loss += float(loss.item())
            batch_count += 1
    avg_loss = total_loss / batch_count
    return avg_loss
