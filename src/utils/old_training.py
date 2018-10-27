import argparse
import json
import random
import os
import os.path as op
import torch
from datetime import datetime
from torch.autograd import Variable

DEFAULT_PRINT_EVERY = 400
STOCHASTIC_SAMPLE_SIZE = 30

################################################################################
# Loss Calculation Functions
################################################################################
def compute_avg_loss(net, loss_fn, loader):
    total_loss = 0.0
    batch_count = 0
    batch_groups = list(loader)
    random.shuffle(batch_groups)
    for seq_batch, target_batch in batch_groups[:STOCHASTIC_SAMPLE_SIZE]:  # stochastic check?
        inpt = Variable(torch.LongTensor(seq_batch))
        target = Variable(torch.LongTensor(target_batch))
        if torch.cuda.is_available() and (not net.no_cuda):
            inpt = inpt.cuda()
            target = target.cuda()
        output = net(inpt)[:, -1, :]
        loss = loss_fn(output, target)
        total_loss += float(loss.item())
        batch_count += 1
    avg_loss = total_loss/batch_count
    return avg_loss

def compute_harmony_conditioned_avg_loss(net, loss_fn, batched_chords,
        batched_seqs, batched_targets):
    total_loss = 0.0
    batch_count = 0
    batch_groups = list(zip(batched_chords, batched_seqs, batched_targets))
    random.shuffle(batch_groups)
    for chord_batch, seq_batch, target_batch in batch_groups[:STOCHASTIC_SAMPLE_SIZE]:
        chord_inpt = Variable(torch.FloatTensor(chord_batch))
        seq_inpt = Variable(torch.LongTensor(seq_batch))
        target = Variable(torch.LongTensor(target_batch))
        if torch.cuda.is_available() and (not net.no_cuda):
            chord_inpt = chord_inpt.cuda()
            seq_inpt = seq_inpt.cuda()
            target = target.cuda()
        output = net(chord_inpt, seq_inpt)[:, -1, :]
        loss = loss_fn(output, target)
        total_loss += float(loss.item())
        batch_count += 1
    avg_loss = total_loss/batch_count
    return avg_loss

def compute_harmony_plus_conditioned_avg_loss(net, loss_fn, batched_chords,
        batched_cond_seqs, batched_seqs, batched_targets):
    total_loss = 0.0
    batch_count = 0
    groups = list(zip(batched_chords, batched_cond_seqs, batched_seqs, batched_targets))
    random.shuffle(groups)
    for chord_batch, cond_batch, seq_batch, target_batch in groups[:STOCHASTIC_SAMPLE_SIZE]:
        chord_inpt = Variable(torch.FloatTensor(chord_batch))
        cond_inpt = Variable(torch.LongTensor(cond_batch))
        seq_inpt = Variable(torch.LongTensor(seq_batch))
        target = Variable(torch.LongTensor(target_batch))
        if torch.cuda.is_available() and (not net.no_cuda):
            chord_inpt = chord_inpt.cuda()
            cond_inpt = cond_inpt.cuda()
            seq_inpt = seq_inpt.cuda()
            target = target.cuda()
        output = net(chord_inpt, cond_inpt, seq_inpt)[:, -1, :]
        loss = loss_fn(output, target)
        total_loss += float(loss.item())
        batch_count += 1
    avg_loss = total_loss/batch_count
    return avg_loss

################################################################################
# Training Functions
################################################################################
def train_net(net, loss_fn, optimizer, epochs, batched_train_seqs, 
        batched_train_targets, batched_valid_seqs, batched_valid_targets, 
        writer, print_every=DEFAULT_PRINT_EVERY):
    interrupted = False
    train_losses = []
    valid_losses = []
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    try:
        # write the initial loss
        train_loss = compute_avg_loss(net, loss_fn, batched_train_seqs, 
            batched_train_targets)
        train_losses.append(train_loss)
        valid_loss = compute_avg_loss(net, loss_fn, batched_valid_seqs, 
            batched_valid_targets)
        valid_losses.append(valid_loss)
        write_loss(train_loss, valid_loss, writer, 0)
        # train
        for epoch in range(epochs):
            batch_count = 0
            avg_loss = 0.0
            epoch_loss = 0.0
            batch_groups = list(zip(batched_train_seqs, batched_train_targets))
            random.shuffle(batch_groups)
            for seq_batch, target_batch in batch_groups:
                # get the data, wrap it in a Variable
                seq_batch_var = Variable(torch.LongTensor(seq_batch))
                target_batch_var = Variable(torch.LongTensor(target_batch))
                if torch.cuda.is_available() and (not net.no_cuda):
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
            train_loss = epoch_loss/batch_count
            train_losses.append(train_loss)
            valid_loss = compute_avg_loss(net, loss_fn, batched_valid_seqs, 
                batched_valid_targets)
            valid_losses.append(valid_loss)
            write_loss(train_loss, valid_loss, writer, epoch + 1)
        print('Finished Training')
    except KeyboardInterrupt:
        print('Training Interrupted')
        interrupted = True
    return (net, interrupted, train_losses, valid_losses)

def train_chord_conditioned_net(net, loss_fn, optimizer, epochs, 
        batched_train_chord_seqs, batched_train_seqs, batched_train_targets, 
        batched_valid_chord_seqs, batched_valid_seqs, batched_valid_targets, 
        writer, print_every=DEFAULT_PRINT_EVERY):
    interrupted = False
    train_losses = []
    valid_losses = []
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    try:
        # write initial loss
        train_loss = compute_harmony_conditioned_avg_loss(net, loss_fn, 
            batched_train_chord_seqs, batched_train_seqs, batched_train_targets)
        train_losses.append(train_loss)
        valid_loss = compute_harmony_conditioned_avg_loss(net, loss_fn, 
            batched_valid_chord_seqs, batched_valid_seqs, batched_valid_targets)
        valid_losses.append(valid_loss)
        write_loss(train_loss, valid_loss, writer, 0)
        # train net
        for epoch in range(epochs):
            batch_count = 0
            avg_loss = 0.0
            epoch_loss = 0.0
            batch_groups = list(zip(batched_train_chord_seqs, batched_train_seqs, 
                batched_train_targets))
            random.shuffle(batch_groups)
            for chord_batch, seq_batch, target_batch in batch_groups: 
                # get the data, wrap it in a Variable
                chord_inpt = Variable(torch.FloatTensor(chord_batch))
                seq_inpt = Variable(torch.LongTensor(seq_batch))
                target_inpt = Variable(torch.LongTensor(target_batch))
                if torch.cuda.is_available() and (not net.no_cuda):
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
            train_loss = epoch_loss/batch_count
            train_losses.append(epoch_loss/batch_count)
            valid_loss = compute_harmony_conditioned_avg_loss(net, loss_fn, 
                batched_valid_chord_seqs, batched_valid_seqs, batched_valid_targets)
            valid_losses.append(valid_loss)
            write_loss(train_loss, valid_loss, writer, epoch + 1)
        print('Finished Training')
    except KeyboardInterrupt:
        print('Training Interrupted')
        interrupted = True

    return (net, interrupted, train_losses, valid_losses)

def train_chord_and_inter_conditioned_net(net, loss_fn, optimizer, epochs, 
        batched_train_chord_seqs, batched_train_cond_seqs, batched_train_seqs, 
        batched_train_targets, batched_valid_chord_seqs, batched_valid_cond_seqs, 
        batched_valid_seqs, batched_valid_targets, writer, print_every=DEFAULT_PRINT_EVERY):
    interrupted = False
    train_losses = []
    valid_losses = []
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    try:
        # write initial loss
        train_loss = compute_harmony_plus_conditioned_avg_loss(net, loss_fn, 
            batched_train_chord_seqs, batched_train_cond_seqs, batched_train_seqs, 
            batched_train_targets)
        train_losses.append(train_loss)
        valid_loss = compute_harmony_plus_conditioned_avg_loss(net, loss_fn, 
            batched_valid_chord_seqs, batched_valid_cond_seqs, batched_valid_seqs, 
            batched_valid_targets)
        valid_losses.append(valid_loss)
        write_loss(train_loss, valid_loss, writer, 0)
        # train net
        for epoch in range(epochs):
            batch_count = 0
            avg_loss = 0.0
            epoch_loss = 0.0
            batch_groups = list(zip(batched_train_chord_seqs, batched_train_cond_seqs, 
                batched_train_seqs, batched_train_targets))
            random.shuffle(batch_groups)
            for chord_batch, cond_batch, seq_batch, target_batch in batch_groups:
                # get the data, wrap it in a Variable
                chord_inpt = Variable(torch.FloatTensor(chord_batch))
                cond_inpt = Variable(torch.LongTensor(cond_batch))
                seq_inpt = Variable(torch.LongTensor(seq_batch))
                target_inpt = Variable(torch.LongTensor(target_batch))
                if torch.cuda.is_available() and (not net.no_cuda):
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
            train_loss = epoch_loss/batch_count
            train_losses.append(train_loss)
            valid_loss = compute_harmony_plus_conditioned_avg_loss(net, loss_fn, 
                batched_valid_chord_seqs, batched_valid_cond_seqs, batched_valid_seqs, 
                batched_valid_targets)
            valid_losses.append(valid_loss)
            write_loss(train_loss, valid_loss, writer, epoch + 1)
        print('Finished Training')
    except KeyboardInterrupt:
        print('Training Interrupted')
        interrupted = True

    return (net, interrupted, train_losses, valid_losses)

