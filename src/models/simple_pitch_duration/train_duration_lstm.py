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
(batched_train_seqs, batched_train_targets,
 batched_valid_seqs, batched_valid_targets) = lsdl.get_batched_dur_seqs(
         seq_len=args.seq_len, batch_size=args.batch_size, target_as_vector=False)

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
