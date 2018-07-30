import argparse
import os
import os.path as op
import pickle
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from tensorboardX import SummaryWriter

from model_classes import DurationLSTM
sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from utils.constants import DEFAULT_PRINT_EVERY
from utils.dataloaders import LeadSheetDataLoader
from utils.training import train_net, save_run

run_datetime_str = datetime.now().strftime('%b%d-%y_%H:%M:%S')
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
parser.add_argument('-hd', '--hidden_dim', default=128, type=int,
                    help="size of hidden state.")
parser.add_argument('-od', '--output_dim', default=128, type=int,
                    help="size of output softmax.")
parser.add_argument('-nl', '--num_layers', default=2, type=int,
                    help="number of lstm layers to use.")
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                    help="learning rate for sgd")
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help="momentum for sgd.")
parser.add_argument('-pe', '--print_every', default=DEFAULT_PRINT_EVERY, type=int,
                    help="how often to print the loss during training.")
parser.add_argument('-k', '--keep', action='store_true',
                    help="save information about this run")
args = parser.parse_args()

if args.title != run_datetime_str:
    args.title = '_'.join([run_datetime_str, args.title])
root_dir = str(Path(op.abspath(__file__)).parents[3])
data_dir = op.join(root_dir, "data", "processed", "datasets")
if args.charlie_parker:
    dataset = pickle.load(open(op.join(data_dir, "charlie_parker_dataset.pkl"), "rb"))
    args.title = '_'.join([args.title, 'CP'])
else:
    dataset = pickle.load(open(op.join(data_dir, "dataset.pkl"), "rb"))
    args.title = '_'.join([args.title, 'FULL'])
info_dict.update(vars(args))

lsdl = LeadSheetDataLoader(dataset, args.num_songs)
batch_dict = lsdl.get_batched_dur_seqs(seq_len=args.seq_len, batch_size=args.batch_size)
batched_train_seqs = batch_dict['batched_train_seqs']
batched_train_targets = batch_dict['batched_train_targets']
batched_valid_seqs = batch_dict['batched_valid_seqs']
batched_valid_targets = batch_dict['batched_valid_targets']

net = DurationLSTM(args.input_dict_size, args.embedding_dim, args.hidden_dim,
                   args.output_dim, num_layers=args.num_layers, batch_size=args.batch_size)
if torch.cuda.is_available():
    net.cuda()
params = net.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.NLLLoss()

dirpath = op.join(os.getcwd(), "runs", "duration")
if args.keep:
    dirpath = op.join(dirpath, args.title)
else:
    dirpath = op.join(dirpath, "test_runs", args.title)
writer = SummaryWriter(op.join(dirpath, 'tensorboard'))

net, interrupted, train_losses, valid_losses = train_net(
        net, loss_fn, optimizer, args.epochs, batched_train_seqs, batched_train_targets,
        batched_valid_seqs, batched_valid_targets, writer, args.print_every)

writer.close()
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

save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, net, args.keep)
