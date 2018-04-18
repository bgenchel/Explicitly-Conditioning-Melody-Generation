import argparse
import json
import os.path as op
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from torch.autograd import Variable

from dataloaders import LeadSheetDataLoader
from models import PitchLSTM


run_datetime_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default=run_datetime_str, type=str,
                    help="custom title for run data directory")
parser.add_argument('-n', '--num_songs', default=1000, type=int,
                    help="number of songs to include in training")
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help="number of training epochs")
parser.add_argument('-b', '--batch_size', default=5, type=int,
                    help="number of training epochs")
parser.add_argument('-sl', '--seq_len', default=1, type=int,
                    help="number of previous steps to consider in prediction.")
parser.add_argument('-id', '--input_dict_size', default=128, type=int,
                    help="range of possible input note values.")
parser.add_argument('-ed', '--embedding_dim', default=20, type=int,
                    help="size of note embeddings.")
parser.add_argument('-hd', '--hidden_dim', default=25, type=int,
                    help="size of hidden state.")
parser.add_argument('-od', '--output_dim', default=128, type=int,
                    help="size of output softmax.")
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help="learning rate for sgd")
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help="momentum for sgd.")
parser.add_argument('-k', '--keep', action='store_true',
                    help="save information about this run")
parser.add_argument('-s', '--show', action='store_true',
                    help="show figures as they are generated")
args = parser.parse_args()
info_dict.update(vars(args))

root_dir = str(Path(op.abspath(__file__)).parents[3])
data_dir = op.join(root_dir, "data", "processed", "pkl")
dataset = pickle.load(open(op.join(data_dir, "dataset.pkl"), "rb"))

# pdb.set_trace()
lsdl = LeadSheetDataLoader(dataset)
batched_sets = lsdl.get_batched_pitch_seqs(seq_len=args.seq_len, batch_size=args.batch_size)
batched_harmony_seqs, batched_pitch_seqs, batched_next_pitches = batched_sets

# bhs.shape = num_batches x seqs per batch x seq len x harmony size
harmony_dim = batched_harmony_seqs.shape[-1]
# 0 = rest, the rest are MIDI numbers

net = PitchLSTM(args.input_dict_size, harmony_dim, args.embedding_dim, args.hidden_dim,
                args.output_dim, seq_len=args.seq_len, batch_size=args.batch_size)
params = net.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)

# loss_fn = nn.NLLLoss()
loss_fn = nn.BCELoss()
# loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()

batch_zip = zip(batched_pitch_seqs, batched_harmony_seqs, batched_next_pitches)
batch_groups = [tup for tup in batch_zip]
# pdb.set_trace()
try:
    train_losses = []
    print_every = 5
    print("Beginning Training")
    print("Cuda available: ", torch.cuda.is_available())
    for epoch in range(10): # 10 epochs to start
        batch_count = 0
        avg_loss = 0.0
        epoch_loss = 0.0
        for i, batch_group in enumerate(batch_groups):
            # pdb.set_trace()
            # get the data, wrap it in a Variable
            pitches_inpt = Variable(torch.LongTensor(batch_group[0]))
            harmony_inpt = Variable(torch.FloatTensor(batch_group[1]))
            target_pitch = Variable(torch.FloatTensor(batch_group[2]))
            if torch.cuda.is_available():
                pitches_inpt = pitches_inpt.cuda()
                harmony_inpt = harmony_inpt.cuda()
                target_pitch = target_pitch.cuda()
            # detach hidden state
            # net.repackage_hidden_and_cell()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            output = net(pitches_inpt, harmony_inpt)[:, -1, :]
            # backward + optimize
            loss = loss_fn(output, target_pitch)
            # loss.backward()
            loss.backward(retain_graph=True)
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

json.dump(train_losses, open('losses.json', 'wb'), indent=4)
