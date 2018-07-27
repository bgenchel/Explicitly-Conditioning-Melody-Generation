import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class BaselineLSTM(nn.Module):
    def __init__(self, input_dict_size, embedding_dim, hidden_dim, output_dim, 
            num_layers=2, batch_size=None, dropout=0.5, batch_norm=True, cuda=True, 
            **kwargs):
        super(BaselineLSTM, self).__init__(**kwargs)
        self.input_dict_size = input_dict_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.cuda = cuda

        self.embedding = nn.Embedding(input_dict_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout)
        mid_dim = (hidden_dim + output_dim) // 2
        self.decode1 = nn.Linear(hidden_dim, mid_dim)
        self.batch_norm = nn.BatchNorm1d(mid_dim)
        self.decode2 = nn.Linear(mid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)
        return 

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.FloatTensor(np.zeros([self.num_layers, batch_size, 
            self.hidden_dim])))
        cell = Variable(torch.FloatTensor(np.zeros([self.num_layers, batch_size, 
            self.hidden_dim])))
        if torch.cuda.is_available() and self.cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)
        return

    def repackage_hidden_and_cell(self):
        new_hidden = Variable(self.hidden_and_cell[0].data)
        new_cell = Variable(self.hidden_and_cell[1].data)
        if torch.cuda.is_available() and self.cuda:
            new_hidden = new_hidden.cuda()
            new_cell = new_cell.cuda()
        self.hidden_and_cell = (new_hidden, new_cell)
        return

    def forward(self, data, **kwargs):
        embedded = self.embedding(data)
        lstm_out, self.hidden_and_cell = self.lstm(embedded, self.hidden_and_cell)
        if self.batch_norm:
            decoded = self.decode2(F.relu(self.batch_norm(self.decode1(lstm_out))))
        else:
            decoded = self.decode2(F.relu(self.decode1(lstm_out)))
        output = self.softmax(decoded)
        return output
