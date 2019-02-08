import os.path as op
import sys
import torch
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from model_classes import PitchLSTM, DurationLSTM

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.helpers as hlp
from utils.training import Trainer

run_datetime_str = datetime.now().strftime('%b%d-%y_%H:%M:%S')
args = hlp.get_args(default_title=run_datetime_str)
if args.title != run_datetime_str:
    args.title = '_'.join([run_datetime_str, args.title])
args.run_datetime_str = run_datetime_str

if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)

if args.model == "pitch":
    Model = PitchLSTM
elif args.model == "duration":
    Model = DurationLSTM

model = Model(hidden_dim=args.hidden_dim,
              seq_len=args.seq_len,
              batch_size=args.batch_size,
              dropout=args.dropout,
              batch_norm=args.batch_norm,
              no_cuda=args.no_cuda)

Trainer(model, args).train_model()
