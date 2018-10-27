import os.path as op
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.autograd import Variable

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
import utils.constants as const

torch.manual_seed(1) 

class BaselineLSTM(nn.Module):
    def __init__(self, vocab_size=None, embed_dim=None, hidden_dim=None, output_dim=None, seq_len=None, 
            batch_size=None, dropout=0.5, batch_norm=True, no_cuda=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = const.NUM_RNN_LAYERS
        self.batch_norm = batch_norm
        self.no_cuda = no_cuda

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=self.num_layers, 
                            batch_first=True, dropout=dropout)
        mid_dim = (hidden_dim + output_dim) // 2
        self.decode1 = nn.Linear(hidden_dim, mid_dim)
        self.decode_bn = nn.BatchNorm1d(seq_len)
        self.decode2 = nn.Linear(mid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)

        if torch.cuda.is_available() and (not self.no_cuda):
            self.cuda()

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
        if torch.cuda.is_available() and (not self.no_cuda):
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)

    def repackage_hidden_and_cell(self):
        new_hidden = Variable(self.hidden_and_cell[0].data)
        new_cell = Variable(self.hidden_and_cell[1].data)
        if torch.cuda.is_available() and (not self.no_cuda):
            new_hidden = new_hidden.cuda()
            new_cell = new_cell.cuda()
        self.hidden_and_cell = (new_hidden, new_cell)

    def forward(self, data):
        embedded = self.embedding(data)
        lstm_out, self.hidden_and_cell = self.lstm(embedded, self.hidden_and_cell)
        decoded = self.decode1(lstm_out)
        if self.batch_norm:
            decoded = self.decode_bn(decoded)
        decoded = self.decode2(F.relu(decoded))
        output = self.softmax(decoded)
        return output


class PitchLSTM(BaselineLSTM):
    def __init__(self, **kwargs):
        super().__init__(vocab_size=const.PITCH_DIM, embed_dim=const.PITCH_EMBED_DIM, 
                         output_dim=const.PITCH_DIM, **kwargs)

class DurationLSTM(BaselineLSTM):
    def __init__(self, **kwargs):
        super().__init__(vocab_size=const.DUR_DIM, embed_dim=const.DUR_EMBED_DIM, 
                         output_dim=const.PITCH_DIM, **kwargs)
