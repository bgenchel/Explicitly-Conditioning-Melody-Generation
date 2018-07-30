import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

class ChordandInterConditionedLSTM(nn.Module):
    def __init__(self, input_dict_size, cond_dict_size, chord_dim, 
                 embedding_dim, cond_embedding_dim, hidden_dim, output_dim, 
                 num_layers=2, batch_size=None, dropout=0.5, batch_norm=True,
                 no_cuda=False, **kwargs):
        super(ChordandInterConditionedLSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_lstm_layers
        self.batch_norm = batch_norm
        self.no_cuda = no_cuda

        chord_encoding_dim = (3*chord_dim)//4
        self.chord_fc1 = nn.Linear(chord_dim, chord_encoding_dim)
        self.chord_bn = nn.BatchNorm1d(chord_encoding_dim)
        self.chord_fc2 = nn.Linear(chord_encoding_dim, chord_encoding_dim)
        self.embedding = nn.Embedding(input_dict_size, embedding_dim)
        self.cond_embedding = nn.Embedding(cond_dict_size, cond_embedding_dim)
        self.encoder = nn.Linear(embedding_dim + cond_embedding_dim + chord_encoding_dim, hidden_dim)
        self.encoder_bn = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=self.num_layers, batch_first=True)
        mid_dim = (hidden_dim + output_dim) // 2
        self.decode1 = nn.Linear(hidden_dim, mid_dim)
        self.decode_bn = nn.BatchNorm1d(mid_dim)
        self.decode2 = nn.Linear(mid_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)

        if torch.cuda.is_available() and (not self.no_cuda):
            self.cuda()
        return 

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.FloatTensor(np.zeros([self.num_layers, 
            batch_size, self.hidden_dim])))
        cell = Variable(torch.FloatTensor(np.zeros([self.num_layers, 
            batch_size, self.hidden_dim])))
        if torch.cuda.is_available() and (not self.no_cuda):
            hidden = hidden.cuda()
            cell = cell.cuda()
        self.hidden_and_cell = (hidden, cell)
        return

    def repackage_hidden_and_cell(self):
        new_hidden = Variable(self.hidden_and_cell[0].data)
        new_cell = Variable(self.hidden_and_cell[1].data)
        if torch.cuda.is_available() and (not self.no_cuda):
            new_hidden = new_hidden.cuda()
            new_cell = new_cell.cuda()
        self.hidden_and_cell = (new_hidden, new_cell)
        return

    def forward(self, chords, conds, data):
        if self.batch_norm:
            encoded_chords = self.chord_fc2(F.relu(self.chord_bn(self.chord_fc1(chords))))
        else:
            encoded_chords = self.chord_fc2(F.relu(self.chord_fc1(chords)))

        embedded_data = self.embedding(data)
        embedded_conds = self.cond_embedding(conds)
        inpt = torch.cat([encoded_chords, embedded_conds, embedded_data], 2) # Concatenate along 3rd dimension
        if self.batch_norm:
            encoded_inpt = F.relu(self.encoder_bn(self.encoder(inpt)))
        else:
            encoded_inpt = F.relu(self.encoder(inpt))

        lstm_out, self.hidden_and_cell = self.lstm(encoded_inpt, self.hidden_and_cell)
        if self.batch_norm:
            decoded = self.decode2(F.relu(self.decode_bn(self.decode1(lstm_out))))
        else:
            decoded = self.decode2(F.relu(self.decode1(lstm_out)))
        output = self.softmax(decoded)
        return output
