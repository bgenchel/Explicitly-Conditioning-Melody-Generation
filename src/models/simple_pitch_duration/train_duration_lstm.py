import argparse
import json
import os
import os.path as op
# import pdb
import pickle
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
# from torch.autograd import Variable

from dataloaders import LeadSheetDataLoader
from models import DurationLSTM
from utils import train_net, save_run

run_datetime_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default=run_datetime_str, type=str,
                    help="custom title for run data directory")
parser.add_argument('-cp', '--charlie_parker', action="store_true",
                    help="use the charlie parker data subset")
parser.add_argument('-n', '--num_songs', default=None, type=int,
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
parser.add_argument('-nl', '--num_layers', default=2, type=int,
                    help="number of lstm layers to use.")
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
data_dir = op.join(root_dir, "data", "processed", "datasets")
if args.charlie_parker:
    dataset = pickle.load(open(op.join(data_dir, "charlie_parker_dataset.pkl"), "rb"))
else:
    dataset = pickle.load(open(op.join(data_dir, "dataset.pkl"), "rb"))

lsdl = LeadSheetDataLoader(dataset, args.num_songs)
batched_sets = lsdl.get_batched_dur_seqs(seq_len=args.seq_len,
                                           batch_size=args.batch_size, 
                                           target_as_vector=False)
batched_train_seqs, batched_train_targets, batched_valid_seqs, batched_valid_targets = batched_sets

# bhs.shape = num_batches x seqs per batch x seq len x harmony size
net = DurationLSTM(args.input_dict_size, args.embedding_dim, args.hidden_dim,
                   args.output_dim, num_layers=args.num_layers, batch_size=args.batch_size)
if torch.cuda.is_available():
    net.cuda()
params = net.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.NLLLoss()
# loss_fn = nn.BCELoss()
# loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()

net, interrupted, train_losses, valid_losses = train_net(
        net, loss_fn, optimizer, args.epochs, batched_train_seqs, batched_train_targets,
        batched_valid_seqs, batched_valid_targets)
# interrupted = False
# train_losses = []
# valid_losses = []
# print_every = 5
# print("Beginning Training")
# print("Cuda available: ", torch.cuda.is_available())
# try:
#     for epoch in range(args.epochs): # 10 epochs to start
#         batch_count = 0
#         avg_loss = 0.0
#         epoch_loss = 0.0
#         for seq_batch, target_batch in zip(batched_train_seqs, batched_train_targets):
#             # get the data, wrap it in a Variable
#             seq_batch_var = Variable(torch.LongTensor(seq_batch))
#             target_batch_var = Variable(torch.LongTensor(target_batch))
#             if torch.cuda.is_available():
#                 seq_batch_var = seq_batch_var.cuda()
#                 target_batch_var = target_batch_var.cuda()
#             # detach hidden state
#             net.repackage_hidden_and_cell()
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward pass
#             output = net(seq_batch_var)[:, -1, :]
#             # backward + optimize
#             loss = loss_fn(output, target_batch_var)
#             loss.backward()
#             optimizer.step()
#             # print stats out
#             avg_loss += loss.data[0]
#             epoch_loss += loss.data[0]
#             if batch_count % print_every == print_every - 1:
#                 print('epoch: %d, batch_count: %d, loss: %.5f'%(
#                     epoch + 1, batch_count + 1, avg_loss / print_every))
#                 avg_loss = 0.0
#             batch_count += 1
#         print('Average Epoch Loss: %f'%(epoch_loss/batch_count))
#         train_losses.append(epoch_loss/batch_count)
#         valid_loss = compute_avg_loss(net, loss_fn, batched_valid_seqs,
#                                       batched_valid_targets)
#         valid_losses.append(valid_loss)
#     print('Finished Training')
# except KeyboardInterrupt:
#     print('Training Interrupted')
#     interrupted = True

info_dict['interrupted'] = interrupted
info_dict['epochs_completed'] = len(train_losses)
info_dict['final_training_loss'] = train_losses[-1]
info_dict['final_valid_loss'] = valid_losses[-1]

model_inputs = {'input_dict_size': args.input_dict_size, 
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'output_dim': args.output_dim,
                'num_layers': args.num_layers,
                'batch_size': args.batch_size}

dirpath = op.join(os.getcwd(), "runs", "durations", args.title)

if args.keep:
    save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, net)
