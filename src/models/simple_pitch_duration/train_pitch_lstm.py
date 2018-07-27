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

from model_classes import BaselineLSTM
sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from utils import training
from utils.constants import PITCH_DIM, DEFAULT_PRINT_EVERY
from utils.dataloaders import LeadSheetDataLoader

torch.cuda.device(0)

run_datetime_str = datetime.now().strftime('%b%d-%y_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

args = training.get_args()
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
batch_dict = lsdl.get_batched_pitch_seqs(seq_len=args.seq_len, batch_size=args.batch_size)
batched_train_seqs = batch_dict['batched_train_seqs']
batched_train_targets = batch_dict['batched_train_targets']
batched_valid_seqs = batch_dict['batched_valid_seqs']
batched_valid_targets = batch_dict['batched_valid_targets']

net = BaselineLSTM(input_dict_size=PITCH_DIM, 
                   embedding_dim=args.pitch_embedding_dim, 
                   hidden_dim=args.hidden_dim,
                   output_dim=PITCH_DIM, 
                   num_layers=args.num_layers, 
                   batch_size=args.batch_size,
                   dropout=args.drop_out,
                   batch_norm=args.batch_norm,
                   cuda=(not args.no_cuda))
if torch.cuda.is_available() and (not args.not_cuda):
    net.cuda()
params = net.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.NLLLoss()

dirpath = op.join(os.getcwd(), "runs", "pitch")
if args.keep:
    dirpath = op.join(dirpath, args.title)
else:
    dirpath = op.join(dirpath, "test_runs", args.title)
writer = SummaryWriter(op.join(dirpath, 'tensorboard'))

net, interrupted, train_losses, valid_losses = training.train_net(net, loss_fn, optimizer, 
        args.epochs, batched_train_seqs, batched_train_targets, batched_valid_seqs, 
        batched_valid_targets, writer, args.print_every)

writer.close()
info_dict['interrupted'] = interrupted
info_dict['epochs_completed'] = len(train_losses)
info_dict['final_training_loss'] = train_losses[-1]
info_dict['final_valid_loss'] = valid_losses[-1]

model_inputs = {'input_dict_size': PITCH_DIM, 
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'output_dim': PITCH_DIM,
                'num_layers': args.num_layers,
                'batch_size': args.batch_size,
                'dropout': args.dropout,
                'batch_norm': args.batch_norm}

training.save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, net, args.keep)
