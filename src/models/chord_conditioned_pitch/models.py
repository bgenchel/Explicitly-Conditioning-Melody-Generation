import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class PitchLSTM(nn.Module):
    def __init__(self, input_dict_size, harmony_dim, embedding_dim, hidden_dim, 
                 output_dim, num_layers=2, batch_size=None, **kwargs):
        super(PitchLSTM, self).__init__(**kwargs)
        self.input_dict_size = input_dict_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        harmony_encoding_dim = (3*harmony_dim)//4

        self.harmony_fc1 = nn.Linear(harmony_dim, harmony_encoding_dim)
        self.harmony_fc2 = nn.Linear(harmony_encoding_dim, harmony_encoding_dim)
        self.pitch_embedding = nn.Embedding(input_dict_size, embedding_dim)
        self.encoder = nn.Linear(embedding_dim + harmony_encoding_dim, hidden_dim)
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

    def forward(self, harmonies, pitches):
        # pdb.set_trace()
        encoded_harmonies = self.harmony_fc2(F.relu(self.harmony_fc1(harmonies)))
        embedded_pitches = self.pitch_embedding(pitches)
        inpt = torch.cat([encoded_harmonies, embedded_pitches], 2) # Concatenate along 3rd dimension
        encoded_inpt = F.relu(self.encoder(inpt))
        lstm_out, self.hidden_and_cell = self.lstm(encoded_inpt, self.hidden_and_cell)
        decoded = self.decoder(lstm_out)
        # num_batches, seq_len, num_feats = decoded.size()
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
