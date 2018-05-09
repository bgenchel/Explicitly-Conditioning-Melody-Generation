import json
import os
import os.path as op
import torch
from torch.autograd import Variable

def compute_avg_loss(net, loss_fn, batched_seqs, batched_targets):
    assert len(batched_seqs) == len(batched_targets)
    total_loss = 0.0
    for seq_batch, target_batch in zip(batched_seqs, batched_targets):
        inpt = Variable(torch.LongTensor(seq_batch))
        target = Variable(torch.LongTensor(target_batch))
        if torch.cuda.is_available():
            inpt = inpt.cuda()
            target = target.cuda()
        output = net(inpt)[:, -1, :]
        loss = loss_fn(output, target)
        total_loss += float(loss.item())
    avg_loss = total_loss/len(batched_seqs)
    return avg_loss

def compute_harmony_conditioned_avg_loss(net, loss_fn, batched_chords,
                                         batched_seqs, batched_targets):
    # import pdb
    # pdb.set_trace()
    assert len(batched_chords) == len(batched_seqs) == len(batched_targets)
    total_loss = 0.0
    for chord_batch, seq_batch, target_batch in zip(batched_chords, batched_seqs, 
                                                    batched_targets):
        chord_inpt = Variable(torch.FloatTensor(chord_batch))
        seq_inpt = Variable(torch.LongTensor(seq_batch))
        target = Variable(torch.LongTensor(target_batch))
        if torch.cuda.is_available():
            chord_inpt = chord_inpt.cuda()
            seq_inpt = seq_inpt.cuda()
            target = target.cuda()
        output = net(chord_inpt, seq_inpt)[:, -1, :]
        loss = loss_fn(output, target)
        total_loss += float(loss.item())
    avg_loss = total_loss/len(batched_seqs)
    return avg_loss

def compute_harmony_plus_conditioned_avg_loss(net, loss_fn, batched_chords,
        batched_cond_seqs, batched_seqs, batched_targets):
    # import pdb
    # pdb.set_trace()
    assert len(batched_chords) == len(batched_cond_seqs) == len(batched_seqs) == len(batched_targets)
    total_loss = 0.0
    for chord_batch, cond_batch, seq_batch, target_batch in zip(
            batched_chords, batched_cond_seqs, batched_seqs, batched_targets):
        chord_inpt = Variable(torch.FloatTensor(chord_batch))
        cond_inpt = Variable(torch.LongTensor(cond_batch))
        seq_inpt = Variable(torch.LongTensor(seq_batch))
        target = Variable(torch.LongTensor(target_batch))
        if torch.cuda.is_available():
            chord_inpt = chord_inpt.cuda()
            cond_inpt = cond_inpt.cuda()
            seq_inpt = seq_inpt.cuda()
            target = target.cuda()
        output = net(chord_inpt, cond_inpt, seq_inpt)[:, -1, :]
        loss = loss_fn(output, target)
        total_loss += float(loss.item())
    avg_loss = total_loss/len(batched_seqs)
    return avg_loss

def train_net(net, loss_fn, optimizer, epochs, batched_train_seqs, batched_train_targets, 
              batched_valid_seqs, batched_valid_targets, print_every=5):
    interrupted = False
    train_losses = []
    valid_losses = []
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    try:
        for epoch in range(epochs): # 10 epochs to start
            batch_count = 0
            avg_loss = 0.0
            epoch_loss = 0.0
            for seq_batch, target_batch in zip(batched_train_seqs, batched_train_targets):
                # get the data, wrap it in a Variable
                seq_batch_var = Variable(torch.LongTensor(seq_batch))
                target_batch_var = Variable(torch.LongTensor(target_batch))
                if torch.cuda.is_available():
                    seq_batch_var = seq_batch_var.cuda()
                    target_batch_var = target_batch_var.cuda()
                # detach hidden state
                net.repackage_hidden_and_cell()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward pass
                output = net(seq_batch_var)[:, -1, :]
                # backward + optimize
                loss = loss_fn(output, target_batch_var)
                loss.backward()
                optimizer.step()
                # print stats out
                avg_loss += float(loss.item())
                epoch_loss += float(loss.item())
                if batch_count % print_every == print_every - 1:
                    print('epoch: %d, batch_count: %d, loss: %.5f'%(
                        epoch + 1, batch_count + 1, avg_loss / print_every))
                    avg_loss = 0.0
                batch_count += 1
            print('Average Epoch Loss: %f'%(epoch_loss/batch_count))
            train_losses.append(epoch_loss/batch_count)
            valid_loss = compute_avg_loss(net, loss_fn, batched_valid_seqs,
                                          batched_valid_targets)
            valid_losses.append(valid_loss)
        print('Finished Training')
    except KeyboardInterrupt:
        print('Training Interrupted')
        interrupted = True

    return (net, interrupted, train_losses, valid_losses)

def train_harmony_conditioned_net(net, loss_fn, optimizer, epochs, 
            batched_train_chord_seqs, batched_train_seqs, batched_train_targets, 
            batched_valid_chord_seqs, batched_valid_seqs, batched_valid_targets, 
            print_every=5):
    interrupted = False
    train_losses = []
    valid_losses = []
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    try:
        for epoch in range(epochs): # 10 epochs to start
            batch_count = 0
            avg_loss = 0.0
            epoch_loss = 0.0
            batched_groups = zip(batched_train_chord_seqs, 
                                 batched_train_seqs, batched_train_targets)
            for chord_batch, seq_batch, target_batch in batched_groups:
                # get the data, wrap it in a Variable
                chord_inpt = Variable(torch.FloatTensor(chord_batch))
                seq_inpt = Variable(torch.LongTensor(seq_batch))
                target_inpt = Variable(torch.LongTensor(target_batch))
                if torch.cuda.is_available():
                    chord_inpt = chord_inpt.cuda()
                    seq_inpt = seq_inpt.cuda()
                    target_inpt = target_inpt.cuda()
                # detach hidden state
                net.repackage_hidden_and_cell()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward pass
                output = net(chord_inpt, seq_inpt)[:, -1, :]
                # backward + optimize
                loss = loss_fn(output, target_inpt)
                loss.backward()
                optimizer.step()
                # print stats out
                avg_loss += float(loss.item())
                epoch_loss += float(loss.item())
                if batch_count % print_every == print_every - 1:
                    print('epoch: %d, batch_count: %d, loss: %.5f'%(
                        epoch + 1, batch_count + 1, avg_loss / print_every))
                    avg_loss = 0.0
                batch_count += 1
            print('Average Epoch Loss: %f'%(epoch_loss/batch_count))
            train_losses.append(epoch_loss/batch_count)
            # import pdb
            # pdb.set_trace()
            valid_loss = compute_harmony_conditioned_avg_loss(
                    net, loss_fn, batched_valid_chord_seqs, batched_valid_seqs, 
                    batched_valid_targets)
            print('Validation Loss: %f'%(valid_loss))
            valid_losses.append(valid_loss)
        print('Finished Training')
    except KeyboardInterrupt:
        print('Training Interrupted')
        interrupted = True

    return (net, interrupted, train_losses, valid_losses)

def train_harmony_plus_conditioned_net(net, loss_fn, optimizer, epochs, 
        batched_train_chord_seqs, batched_train_cond_seqs, batched_train_seqs, 
        batched_train_targets, batched_valid_chord_seqs, batched_valid_cond_seqs, 
        batched_valid_seqs, batched_valid_targets, print_every=5):
    interrupted = False
    train_losses = []
    valid_losses = []
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    try:
        for epoch in range(epochs): # 10 epochs to start
            batch_count = 0
            avg_loss = 0.0
            epoch_loss = 0.0
            batched_groups = zip(batched_train_chord_seqs, batched_train_cond_seqs,
                                 batched_train_seqs, batched_train_targets)
            for chord_batch, cond_batch, seq_batch, target_batch in batched_groups:
                # get the data, wrap it in a Variable
                chord_inpt = Variable(torch.FloatTensor(chord_batch))
                cond_inpt = Variable(torch.LongTensor(cond_batch))
                seq_inpt = Variable(torch.LongTensor(seq_batch))
                target_inpt = Variable(torch.LongTensor(target_batch))
                if torch.cuda.is_available():
                    chord_inpt = chord_inpt.cuda()
                    cond_inpt = cond_inpt.cuda()
                    seq_inpt = seq_inpt.cuda()
                    target_inpt = target_inpt.cuda()
                # detach hidden state
                net.repackage_hidden_and_cell()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward pass
                output = net(chord_inpt, cond_inpt, seq_inpt)[:, -1, :]
                # backward + optimize
                loss = loss_fn(output, target_inpt)
                loss.backward()
                optimizer.step()
                # print stats out
                avg_loss += float(loss.item())
                epoch_loss += float(loss.item())
                if batch_count % print_every == print_every - 1:
                    print('epoch: %d, batch_count: %d, loss: %.5f'%(
                        epoch + 1, batch_count + 1, avg_loss / print_every))
                    avg_loss = 0.0
                batch_count += 1
            print('Average Epoch Loss: %f'%(epoch_loss/batch_count))
            train_losses.append(epoch_loss/batch_count)
            # import pdb
            # pdb.set_trace()
            valid_loss = compute_harmony_plus_conditioned_avg_loss(
                    net, loss_fn, batched_valid_chord_seqs, batched_valid_cond_seqs,
                    batched_valid_seqs, batched_valid_targets)
            print('Validation Loss: %f'%(valid_loss))
            valid_losses.append(valid_loss)
        print('Finished Training')
    except KeyboardInterrupt:
        print('Training Interrupted')
        interrupted = True

    return (net, interrupted, train_losses, valid_losses)

def save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, model):
    if not op.exists(dirpath):
        os.makedirs(dirpath)

    print('Writing run info file ...')
    with open(op.join(dirpath, 'info.txt'), 'w') as fp:
        max_kw_len = max([len(key) for key in info_dict.keys()])
        for k, v in info_dict.items():
            space_buffer = ' '*(max_kw_len - len(k))
            fp.write('%s:%s\t %s\n'%(str(k), space_buffer, str(v)))
        fp.close()

    print('Writing training losses ...') 
    json.dump(train_losses, open(op.join(dirpath, 'train_losses.json'), 'w'), indent=4)

    print('Writing validation losses ...') 
    json.dump(valid_losses, open(op.join(dirpath, 'valid_losses.json'), 'w'), indent=4)

    print('Saving model ...')
    json.dump(model_inputs, open(op.join(dirpath, 'model_inputs.json'), 'w'), indent=4)
    torch.save(model.state_dict(), op.join(dirpath, 'model_state.pt'))
    return
