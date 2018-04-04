import numpy as np
import os
import os.path as op
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path

from dataloaders import LeadSheetDataLoader
from models import ChordLSTM


root_dir = str(Path(op.abspath(__file__)).parents[3])
data_dir = op.join(root_dir, "data", "processed", "pkl")
dataset = pickle.load(open(op.join(data_dir, "dataset.pkl"), "rb"))

dl = LeadSheetDataLoader(dataset)
batched_harmony_seqs, batched_next_harmonies = dl.get_batched_harmony(batch_size=5)

input_dim = np.prod(batched_harmony_seqs[0].size()[-2:])
net = ChordLSTM(input_dim)

params = net.parameters()
optimizer = optim.Adam(params, lr=.001)

loss_fn = nn.MSELoss(size_average=False)

try:
    train_losses = []
    print_every = 5
    print("Beginning Training")
    for epoch in range(10): # 10 epochs to start
        batch_count = 0
        avg_loss = 0.0
        epoch_loss = 0.0
        for i, harmony_seq_batch in enumerate(batched_harmony_seqs):
            # get the data, wrap it in a Variable
            inpt = Variable(torch.FloatTensor(harmony_seq_batch))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            output = net(inpt)
            # backward + optimize
            next_harmony_batch = batched_next_harmonies[i]
            loss = loss_fn(output, next_harmony_batch)
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
