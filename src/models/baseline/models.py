import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

# CHORD_DIM = 12
# NUM_PITCH_CLASSES = 13
# NUM_OCTAVES = 9
# NUM_DURATION_CLASSES = 18

# class HarmonyLSTM(nn.Module):
#     def __init__(self, input_dim, bidirectional=False, **kwargs):
#         super(HarmonyLSTM, self).__init__(**kwargs)
#         self.input_dim = input_dim
#         self.hidden_dim = (input_dim*2)//3
#         self.lstm = nn.LSTM(input_dim, (input_dim*2)//3, num_layers=2,
#                             batch_first=True, bidirectional=bidirectional)
#         self.fc_out = nn.Linear(hidden_dim, input_dim)
#         self.sigmoid = nn.Sigmoid()

#         self.cell_and_hidden = self.init_cell_and_hidden()
#         return 

#     def init_cell_and_hidden(self):
#         return (Variable(torch.FloatTensor(1, 1, self.hidden_dim).normal_()),
#                 Variable(torch.FloatTensor(1, 1, self.hidden_dim).normal_()))

#     def forward(self, inpt):
#         chord_seq = inpt.view(inpt.size()[0], np.prod(inpt.size()[1:]))
#         lstm_out, self.cell_and_hidden = self.lstm(chord_seq, self.cell_and_hidden)
#         out = self.sigmoid(self.fc_out(lstm_out))
#         return out


class PitchLSTM(nn.Module):
    def __init__(self, input_dim, harmony_dim, embedding_dim, hidden_dim, 
                 output_dim, num_layers=2, seq_len=2, batch_size=None, **kwargs):
        super(PitchLSTM, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.harmony_dim = harmony_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.harmony_encoder = nn.Linear(harmony_dim, harmony_dim)
        self.pitch_embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder = nn.Linear(embedding_dim + harmony_dim, hidden_dim)
        self.lstm = nn.LSTM(seq_len*hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax()

        self.hidden_and_cell = None
        if batch_size is not None:
            self.init_hidden_and_cell(batch_size)
        return 

    def init_hidden_and_cell(self, batch_size):
        self.hidden_and_cell = (
                Variable(torch.FloatTensor(self.num_layers, batch_size, self.hidden_dim).normal_()),
                Variable(torch.FloatTensor(self.num_layers, batch_size, self.hidden_dim).normal_()))
        return

    def forward(self, pitches, harmonies):
        pdb.set_trace()
        encoded_harmonies = self.harmony_encoder(harmonies)
        embedded_pitches = self.pitch_embedding(pitches)
        inpt = torch.cat([encoded_harmonies, embedded_pitches], 2) # Concatenate along 3rd dimension
        encoded_inpt = F.relu(self.encoder(inpt))
        encoded_inpt = encoded_inpt.view((int(np.prod(encoded_inpt.shape[:-1])), -1))
        lstm_out, self.hidden_and_cell = self.lstm(encoded_inpt, self.hidden_and_cell)
        decoded = self.decoder(lstm_out)
        num_batches, seq_len, num_feats = decoded.size()
        output = self.softmax(decoded.view(num_batches, seq_len*num_feats))
        return output


# class DurationNet(nn.Module):
#     def __init__(self, chord_dim, embedding_dim, **kwargs):
#         super(DeepJazzClone, self).__init__(**kwargs)
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



