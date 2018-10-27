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

from model_classes import PitchLSTM, DurationLSTM
from training import train_epoch, eval_epoch
sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const
from utils.data.datasets import BebopPitchDurDataset
from utils.data.dataloaders import SplitDataLoader
from utils import training as tr

torch.cuda.device(0)

run_datetime_str = datetime.now().strftime('%b%d-%y_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

args = tr.get_args(default_title=run_datetime_str)
if args.title != run_datetime_str:
    args.title = '_'.join([run_datetime_str, args.title])
info_dict.update(vars(args))

dataset = BebopPitchDurDataset(seq_len=args.seq_len)
train_loader, valid_loader = SplitDataLoader(dataset, batch_size=args.batch_size, drop_last=True).split()

if args.model == "pitch":
    Model = PitchLSTM
    KEY = const.PITCH_KEY
elif args.model == "duration":
    Model = DurationLSTM
    KEY = const.DUR_KEY

model = Model(hidden_dim=args.hidden_dim,
              seq_len=args.seq_len,
              batch_size=args.batch_size,
              dropout=args.dropout,
              batch_norm=args.batch_norm,
              no_cuda=args.no_cuda)

params = model.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.NLLLoss()

def assembler(data_dict):
    data = data_dict[KEY]
    if torch.cuda.is_available() and (not args.no_cuda):
        data = data.cuda()
    return data

dirpath = op.join(os.getcwd(), "runs", args.model)
if args.keep:
    dirpath = op.join(dirpath, args.title)
else:
    dirpath = op.join(dirpath, "test_runs", args.title)
writer = SummaryWriter(op.join(dirpath, 'tensorboard'))

interrupted = False
train_losses, valid_losses = [], []
print("Beginning Training - %s model" % args.model)
print("Cuda available: ", torch.cuda.is_available())
try:
    # write the initial loss
    init_train_loss = eval_epoch(model, train_loader, assembler, loss_fn)  
    train_losses.append(init_train_loss)
    print("Initial Training Loss: %.5f" % (init_train_loss))
    init_valid_loss = eval_epoch(model, valid_loader, assembler, loss_fn)
    valid_losses.append(init_valid_loss)
    print("Initial Validation Loss: %.5f" % (init_valid_loss))
    tr.write_loss(train_losses[0], valid_losses[0], writer, 0)
    # train
    for epoch in range(args.epochs):
        print("="*20 + "\nEpoch %d / %d\n" % (epoch + 1, args.epochs) + "="*20)
        train_loss = train_epoch(model, train_loader, assembler, loss_fn, optimizer)
        train_losses.append(train_loss)
        print("Epoch %d Training Loss: %.5f" % (epoch + 1, train_loss))
        valid_loss = eval_epoch(model, valid_loader, assembler, loss_fn)
        valid_losses.append(valid_loss)
        print("Epoch %d Validation Loss: %.5f" % (epoch + 1, valid_loss))
        tr.write_loss(train_loss, valid_loss, writer, epoch + 1)
    print("Finished Training.")
except KeyboardInterrupt:
    print("Training Interrupted.")
    interrupted = True

writer.close()
info_dict['interrupted'] = interrupted
info_dict['epochs_completed'] = len(train_losses)
info_dict['final_tr.loss'] = train_losses[-1]
info_dict['final_valid_loss'] = valid_losses[-1]

model_inputs = {'hidden_dim': args.hidden_dim,
                'seq_len': args.seq_len,
                'batch_size': args.batch_size,
                'dropout': args.dropout,
                'batch_norm': args.batch_norm,
                'no_cuda': args.no_cuda}

tr.save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, model, args.keep)
