import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class PitchLSTM(nn.Module):
    def __init__(self, input_dict_size, embedding_dim, hidden_dim, 
                 output_dim, num_layers=2, batch_size=None, **kwargs):
        super(PitchLSTM, self).__init__(**kwargs)
        self.input_dict_size = input_dict_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.pitch_embedding = nn.Embedding(input_dict_size, embedding_dim)
        # self.encoder = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
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

    def forward(self, pitches):
        embedded_pitches = self.pitch_embedding(pitches)
        lstm_out, self.hidden_and_cell = self.lstm(embedded_pitches, self.hidden_and_cell)
        decoded = self.decoder(lstm_out)
        output = self.softmax(decoded)
        return output


class DurationLSTM(nn.Module):
    def __init__(self, input_dict_size, embedding_dim, hidden_dim, 
                 output_dim, num_layers=2, batch_size=None, **kwargs):
        super(DurationLSTM, self).__init__(**kwargs)
        self.input_dict_size = input_dict_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.dur_embedding = nn.Embedding(input_dict_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
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

    def forward(self, pitches):
        embedded_pitches = self.dur_embedding(pitches)
        lstm_out, self.hidden_and_cell = self.lstm(embedded_pitches, self.hidden_and_cell)
        decoded = self.decoder(lstm_out)
        output = self.softmax(decoded)
        return output

# class DurationLSTM(nn.Module):
#     def __init__(self, chord_dim, embedding_dim, **kwargs):
#         super(DurationLSTM, self).__init__(**kwargs)
#         self.fc_chord_input = nn.Linear(chord_dim, embedding_dim)
#         self.pitch_class_embeddings = nn.Embedding(NUM_PITCH_CLASSES, embedding_dim)
#         self.octave_embeddings = nn.Embedding(NUM_DURATION_CLASSES, embedding_dim)
#         self.lstm = nn.LSTM(
#         # self.fc1 = nn.Linear(input_dim, input_dim/2)
#         # self.fc2 = nn.Linear(input_dim/2, input_dim/4)
#         # self.fc3 = nn.Linear(input_dim/4, input_dim/8)
#         return 

#     def forward(self, inpt):
#         inpt = inpt.view(inpt.size()[0], np.prod(inpt.size()[1:]))
#         enc = F.relu(self.fc1(inpt))
#         enc = F.relu(self.fc2(enc))
#         enc = F.relu(self.fc3(enc))
#         return enc



