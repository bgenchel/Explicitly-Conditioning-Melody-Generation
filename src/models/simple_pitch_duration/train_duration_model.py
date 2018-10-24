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
from training import train_epoch, eval_epoch
sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const
from utils.data.datasets import BebopPitchDurDataset
from utils.data.dataloaders import SplitDataLoader
from utils import training

torch.cuda.device(0)

run_datetime_str = datetime.now().strftime('%b%d-%y_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

args = training.get_args(default_title=run_datetime_str)
if args.title != run_datetime_str:
    args.title = '_'.join([run_datetime_str, args.title])
info_dict.update(vars(args))

dataset = BebopPitchDurDataset(seq_len=args.seq_len)
train_loader, valid_loader = SplitDataLoader(dataset, batch_size=args.batch_size).split()

model = PitchLSTM(hidden_dim=args.hidden_dim,
                  seq_len=args.seq_len,
                  batch_size=args.batch_size,
                  dropout=args.dropout,
                  batch_norm=args.batch_norm,
                  no_cuda=args.no_cuda)

params = model.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.NLLLoss()

def assembler(data_dict):
    data = data_dict["duration_tags"]
    if torch.cuda.is_available() and (not args.no_cuda):
        data = data.cuda()
    return data

dirpath = op.join(os.getcwd(), "runs", "duration")
if args.keep:
    dirpath = op.join(dirpath, args.title)
else:
    dirpath = op.join(dirpath, "test_runs", args.title)
writer = SummaryWriter(op.join(dirpath, 'tensorboard'))

interrupted = False
train_losses, valid_losses = [], []
print("Beginning Training")
print("Cuda available: ", torch.cuda.is_available())
try:
    # write the initial loss
    init_train_loss = eval_epoch(model, train_loader, assembler, loss_fn, desc=" - Eval Initial Training Loss")  
    init_valid_loss = eval_epoch(model, valid_loader, assembler, loss_fn, desc=" - Eval Initial Validation Loss")
    train_losses.append(init_train_loss)
    valid_losses.append(init_valid_loss)
    write_loss(train_losses[0], valid_losses[0], writer, 0)
    # train
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, assembler, loss_fn, optimizer, 
                                 desc=" - Training - Epoch %d / %d"%(epoch, args.epochs))
        train_losses.append(train_loss)
        valid_loss = eval_epoch(model, valid_loader, assembler, loss_fn)
        valid_losses.append(valid_loss)
        write_loss(train_loss, valid_loss, writer, epoch + 1)
    print('Finished Training')
except KeyboardInterrupt:
    print('Training Interrupted')
    interrupted = True

writer.close()
info_dict['interrupted'] = interrupted
info_dict['epochs_completed'] = len(train_losses)
info_dict['final_training_loss'] = train_losses[-1]
info_dict['final_valid_loss'] = valid_losses[-1]

model_inputs = {'hidden_dim': args.hidden_dim,
                'seq_len': args.seq_len,
                'batch_size': args.batch_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'batch_norm': args.batch_norm,
                'no_cuda': args.no_cuda}

training.save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, net, args.keep)
