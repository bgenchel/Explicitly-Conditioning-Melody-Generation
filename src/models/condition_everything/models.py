import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class PitchLSTM(nn.Module):
    def __init__(self, pitch_dict_size, dur_dict_size, harmony_dim, pitch_embedding_dim, 
                 dur_embedding_dim, hidden_dim, output_dim, num_layers=2, 
                 batch_size=None, **kwargs):
        super(PitchLSTM, self).__init__(**kwargs)
        self.input_dict_size = input_dict_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        harmony_encoding_dim = (3*harmony_dim)//4

        self.harmony_fc1 = nn.Linear(harmony_dim, harmony_encoding_dim)
        self.harmony_fc2 = nn.Linear(harmony_encoding_dim, harmony_encoding_dim)
        self.pitch_embedding = nn.Embedding(pitch_dict_size, pitch_embedding_dim)
        self.dur_embedding = nn.Embedding(dur_dict_size, dur_embedding_dim)
        self.encoder = nn.Linear(pitch_embedding_dim + dur_embedding_dim + harmony_encoding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)
        return 

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.FloatTensor(self.num_layers, batch_size, self.hidden_dim).normal_())
        cell = Variable(torch.FloatTensor(self.num_layers, batch_size, self.hidden_dim).normal_())
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)
        return

    def repackage_hidden_and_cell(self):
        new_hidden = Variable(self.hidden_and_cell[0].data)
        new_cell = Variable(self.hidden_and_cell[1].data)
        if torch.cuda.is_available():
            new_hidden = new_hidden.cuda()
            new_cell = new_cell.cuda()
        self.hidden_and_cell = (new_hidden, new_cell)
        return

    def forward(self, pitches, durs, harmonies):
        encoded_harmonies = self.harmony_fc2(F.relu(self.harmony_fc1(harmonies)))
        embedded_pitches = self.pitch_embedding(pitches)
        embedded_durs = self.dur_embedding(durs)
        inpt = torch.cat([encoded_harmonies, embedded_durs, embedded_pitches], 2) # Concatenate along 3rd dimension
        encoded_inpt = F.relu(self.encoder(inpt))
        lstm_out, self.hidden_and_cell = self.lstm(encoded_inpt, self.hidden_and_cell)
        decoded = self.decoder(lstm_out)
        # num_batches, seq_len, num_feats = decoded.size()
        output = self.softmax(decoded)
        return output


class DurationLSTM(nn.Module):
    def __init__(self, dur_dict_size, pitch_dict_size, harmony_dim, dur_embedding_dim, 
                 pitch_embedding_dim, hidden_dim, output_dim, num_layers=2, 
                 batch_size=None, **kwargs):
        super(DurationLSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        harmony_encoding_dim = (3*harmony_dim)//4

        self.harmony_fc1 = nn.Linear(harmony_dim, harmony_encoding_dim)
        self.harmony_fc2 = nn.Linear(harmony_encoding_dim, harmony_encoding_dim)

        self.dur_embedding = nn.Embedding(dur_dict_size, dur_embedding_dim)
        self.pitch_embedding = nn.Embedding(pitch_dict_size, pitch_embedding_dim)
        self.encoder = nn.Linear(dur_embedding_dim + pitch_embedding_dim + harmony_encoding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)
        return 

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.FloatTensor(self.num_layers, batch_size, self.hidden_dim).normal_())
        cell = Variable(torch.FloatTensor(self.num_layers, batch_size, self.hidden_dim).normal_())
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)
        return

    def repackage_hidden_and_cell(self):
        new_hidden = Variable(self.hidden_and_cell[0].data)
        new_cell = Variable(self.hidden_and_cell[1].data)
        if torch.cuda.is_available():
            new_hidden = new_hidden.cuda()
            new_cell = new_cell.cuda()
        self.hidden_and_cell = (new_hidden, new_cell)
        return

    def forward(self, durs, pitches, harmonies):
        encoded_harmonies = self.harmony_fc2(F.relu(self.harmony_fc1(harmonies)))
        embedded_durs = self.dur_embedding(durs)
        embedded_pitches = self.pitch_embedding(pitches)
        inpt = torch.cat([encoded_harmonies, embedded_pitches, embedded_durs], 2) # Concatenate along 3rd dimension
        encoded_inpt = F.relu(self.encoder(inpt))
        lstm_out, self.hidden_and_cell = self.lstm(encoded_inpt, self.hidden_and_cell)
        decoded = self.decoder(lstm_out)
        # num_batches, seq_len, num_feats = decoded.size()
        output = self.softmax(decoded)
        return output
