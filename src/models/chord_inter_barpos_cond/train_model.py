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

from model_classes import PitchLSTM, DurationLSTM

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const
import utils.helpers as hlp
import utils.training as trn
from utils.data.datasets import BebopPitchDurDataset
from utils.data.dataloaders import SplitDataLoader
from utils.logging import TensorBoardWriter

torch.cuda.device(0)

run_datetime_str = datetime.now().strftime('%b%d-%y_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

args = hlp.get_args(default_title=run_datetime_str)
if args.title != run_datetime_str:
    args.title = '_'.join([run_datetime_str, args.title])
info_dict.update(vars(args))

dataset = BebopPitchDurDataset(seq_len=args.seq_len)
train_loader, valid_loader = SplitDataLoader(dataset, batch_size=args.batch_size, drop_last=True).split()

if args.model == "pitch":
    Model = PitchLSTM
    X_KEY = const.PITCH_KEY
    COND_KEY = const.DUR_KEY
elif args.model == "duration":
    Model = DurationLSTM
    X_KEY = const.DUR_KEY
    COND_KEY = const.PITCH_KEY

def data_assembler(data_dict):
    x = data_dict[X_KEY]
    cond = data_dict[COND_KEY]
    harmony = data_dict[const.CHORD_KEY]
    barpos = data_dict[const.BARPOS_DIM]
    if torch.cuda.is_available() and (not args.no_cuda):
        data = data.cuda()
        cond = cond.cuda()
        harmony = harmony.cuda()
    return (x, cond, harmony)

def target_assembler(target_dict):
    target = target_dict[X_KEY]
    if torch.cuda.is_available() and (not args.no_cuda):
        target = target.cuda()
    return target

model = Model(hidden_dim=args.hidden_dim,
              seq_len=args.seq_len,
              batch_size=args.batch_size,
              dropout=args.dropout,
              batch_norm=args.batch_norm,
              no_cuda=args.no_cuda)

params = model.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)
loss_fn = nn.NLLLoss()

dirpath = op.join(os.getcwd(), "runs", args.model)
if args.keep:
    dirpath = op.join(dirpath, args.title)
else:
    dirpath = op.join(dirpath, "test_runs", args.title)
writer = TensorBoardWriter(op.join(dirpath, 'tensorboard'))

interrupted = False
train_losses, valid_losses = [], []
print("Beginning Training - %s model" % args.model)
print("Cuda available: ", torch.cuda.is_available())
try:
    # write the initial loss
    init_train_loss = trn.eval_epoch(model, train_loader, data_assembler, target_assembler, loss_fn)  
    train_losses.append(init_train_loss)
    print("Initial Training Loss: %.5f" % (init_train_loss))
    init_valid_loss = trn.eval_epoch(model, valid_loader, data_assembler, target_assembler, loss_fn)
    valid_losses.append(init_valid_loss)
    print("Initial Validation Loss: %.5f" % (init_valid_loss))
    writer.write_loss({'training': init_train_loss, 'validation': init_valid_loss}, 0)
    # train
    for epoch in range(args.epochs):
        epoch_label = epoch + 1
        print("="*20 + "\nEpoch %d / %d\n" % (epoch_label, args.epochs) + "="*20)
        train_loss = trn.train_epoch(model, train_loader, data_assembler, target_assembler, loss_fn, optimizer)
        train_losses.append(train_loss)
        print("Epoch %d Training Loss: %.5f" % (epoch_label, train_loss))
        valid_loss = trn.eval_epoch(model, valid_loader, data_assembler, target_assembler, loss_fn)
        valid_losses.append(valid_loss)
        print("Epoch %d Validation Loss: %.5f" % (epoch_label, valid_loss))
        writer.write_loss({'training': train_loss, 'validation': valid_loss}, epoch + 1)
    print("Finished Training.")
except KeyboardInterrupt:
    print("Training Interrupted.")
    interrupted = True

writer.close()
info_dict['interrupted'] = interrupted
info_dict['epochs_completed'] = len(train_losses)
info_dict['final_training_loss'] = train_losses[-1]
info_dict['final_valid_loss'] = valid_losses[-1]

model_inputs = {'hidden_dim': args.hidden_dim,
                'seq_len': args.seq_len,
                'batch_size': args.batch_size, 
                'dropout': args.dropout,
                'batch_norm': args.batch_norm,
                'no_cuda': args.no_cuda}

hlp.save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, model, args.keep)
