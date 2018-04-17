import numpy as np
import os
import os.path as op
import pdb
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path

from dataloaders import LeadSheetDataLoader
from models import PitchLSTM

BATCH_SIZE = 5
SEQ_LEN = 2

root_dir = str(Path(op.abspath(__file__)).parents[3])
data_dir = op.join(root_dir, "data", "processed", "pkl")
dataset = pickle.load(open(op.join(data_dir, "dataset.pkl"), "rb"))

lsdl = LeadSheetDataLoader(dataset)
batched_sets = lsdl.get_batched_pitch_seqs(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
batched_harmony_seqs, batched_pitch_seqs, batched_next_pitches = batched_sets

# bhs.shape = num_batches x seqs per batch x seq len x harmony size
harmony_dim = batched_harmony_seqs.shape[-1]
# 0 = rest, the rest are MIDI numbers
input_dim = 128
# arbitrary values
embedding_dim = 20
hidden_dim = 25
output_dim = 128

net = PitchLSTM(input_dim, harmony_dim, embedding_dim, hidden_dim, output_dim,
                seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
params = net.parameters()
optimizer = optim.Adam(params, lr=.001)
# loss_fn = nn.NLLLoss()
loss_fn = nn.BCELoss()

batch_groups = zip(batched_pitch_seqs, batched_harmony_seqs, batched_next_pitches)
try:
    train_losses = []
    print_every = 5
    print("Beginning Training")
    for epoch in range(10): # 10 epochs to start
        batch_count = 0
        avg_loss = 0.0
        epoch_loss = 0.0
        for i, batch_group in enumerate(batch_groups):
            pdb.set_trace()
            # get the data, wrap it in a Variable
            pitches_inpt = Variable(torch.LongTensor(batch_group[0]))
            harmony_inpt = Variable(torch.FloatTensor(batch_group[1]))
            target_pitch = Variable(torch.LongTensor(batch_group[2]))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            output = net(pitches_inpt, harmony_inpt)
            # backward + optimize
            loss = loss_fn(output, target_pitch)
            loss.backward()
            optimizer.step()
            # print stats out
            avg_loss += loss.data[0]
            epoch_loss += loss.data[0]
            if batch_count % print_every == print_every - 1:
                print('epoch: %d, batch_count: %d, loss: %.5f'%(
                    epoch + 1, batch_count + 1, avg_loss / print_every))
                avg_loss = 0.0
            batch_count += 1
        print('Average Epoch Loss: %f'%(epoch_loss/batch_count))
        train_losses.append(epoch_loss/batch_count)
    print('Finished Training')
except KeyboardInterrupt:
    print('Training Interrupted')
    interrupted = True
