import argparse
import os
import os.path as op
import pdb
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from dataloaders import LeadSheetDataLoader
from models import PitchLSTM
from utils import train_harmony_conditioned_net, save_run

torch.cuda.device(0)

run_datetime_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
info_dict = OrderedDict()
info_dict['run_datetime'] = run_datetime_str

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default=run_datetime_str, type=str,
                    help="custom title for run data directory")
parser.add_argument('-cp', '--charlie_parker', action="store_true",
                    help="use the charlie parker dataset.")
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
# parser.add_argument('-m', '--momentum', default=0.9, type=float,
#                     help="momentum for sgd.")
parser.add_argument('-k', '--keep', action='store_true',
                    help="save information about this run")
args = parser.parse_args()
info_dict.update(vars(args))

root_dir = str(Path(op.abspath(__file__)).parents[3])
data_dir = op.join(root_dir, "data", "processed", "datasets")
if args.charlie_parker:
    dataset = pickle.load(open(op.join(data_dir, "charlie_parker_dataset.pkl"), "rb"))
else:
    dataset = pickle.load(open(op.join(data_dir, "dataset.pkl"), "rb"))

lsdl = LeadSheetDataLoader(dataset, num_songs=args.num_songs)
# (batched_train_pitch_seqs, batched_train_pitch_targets,
#  batched_valid_pitch_seqs, batched_valid_pitch_targets) = lsdl.get_batched_pitch_seqs(
#         seq_len=args.seq_len, batch_size=args.batch_size)
batch_dict = lsdl.get_batched_pitch_seqs(seq_len=args.seq_len, batch_size=args.batch_size)
batched_train_pitch_seqs = batch_dict['batched_train_seqs']
batched_train_pitch_targets = batch_dict['batched_train_targets']
batched_valid_pitch_seqs = batch_dict['batched_valid_seqs']
batched_valid_pitch_targets = batch_dict['batched_valid_targets']

# (batched_train_chord_seqs, batched_train_chord_targets,
#  batched_valid_chord_seqs, batched_valid_chord_seqs) = lsdl.get_batched_harmony(
#         seq_len=args.seq_len, batch_size=args.batch_size)
batch_dict = lsdl.get_batched_harmony(seq_len=args.seq_len, batch_size=args.batch_size)
batched_train_chord_seqs = batch_dict['batched_train_seqs']
batched_train_chord_targets = batch_dict['batched_train_targets']
batched_valid_chord_seqs = batch_dict['batched_valid_seqs']
batched_valid_chord_targets = batch_dict['batched_valid_targets']
# pdb.set_trace()

# batched_harmony_seqs.shape = num_batches x seqs per batch x seq len x harmony size
# 0 = rest, the rest are MIDI numbers
harmony_dim = batched_train_chord_seqs.shape[-1]

net = PitchLSTM(args.input_dict_size, harmony_dim, args.embedding_dim, args.hidden_dim,
                args.output_dim, num_layers=args.num_layers, batch_size=args.batch_size)
if torch.cuda.is_available():
    net.cuda()
params = net.parameters()
optimizer = optim.Adam(params, lr=args.learning_rate)

loss_fn = nn.NLLLoss()
# loss_fn = nn.BCELoss()
# loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()

net, interrupted, train_losses, valid_losses = train_harmony_conditioned_net(
        net, loss_fn, optimizer, args.epochs, batched_train_chord_seqs, 
        batched_train_pitch_seqs, batched_train_pitch_targets, batched_valid_chord_seqs, 
        batched_valid_pitch_seqs, batched_valid_pitch_targets)

# try:
#     interrupted = False
#     train_losses = []
#     print_every = 5
#     print("Beginning Training")
#     print("Cuda available: ", torch.cuda.is_available())
#     for epoch in range(args.epochs): # 10 epochs to start
#         batch_count = 0
#         avg_loss = 0.0
#         epoch_loss = 0.0
#         for i, batch_group in enumerate(batch_groups):
#             # pdb.set_trace()
#             # get the data, wrap it in a Variable
#             harmony_inpt = Variable(torch.FloatTensor(batch_group[0]))
#             pitches_inpt = Variable(torch.LongTensor(batch_group[1]))
#             target_pitch = Variable(torch.LongTensor(batch_group[2]))
#             if torch.cuda.is_available():
#                 harmony_inpt = harmony_inpt.cuda()
#                 pitches_inpt = pitches_inpt.cuda()
#                 target_pitch = target_pitch.cuda()
#             # detach hidden state
#             net.repackage_hidden_and_cell()
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward pass
#             output = net(pitches_inpt, harmony_inpt)[:, -1, :]
#             # backward + optimize
#             loss = loss_fn(output, target_pitch)
#             loss.backward()
#             # loss.backward(retain_graph=True)
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
                'batch_size': args.batch_size, 
                'harmony_dim': harmony_dim}

dirpath = op.join(os.getcwd(), "runs", "pitches", args.title)

if args.keep:
    save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, net)
