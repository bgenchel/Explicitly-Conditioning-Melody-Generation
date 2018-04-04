import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(1)

# CHORD_DIM = 12
# NUM_PITCH_CLASSES = 13
# NUM_OCTAVES = 9
# NUM_DURATION_CLASSES = 18

class ChordLSTM(nn.Module):
    def __init__(self, input_dim, bidirectional=False, **kwargs):
        super(ChordLSTM, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = (input_dim*2)//3
        self.lstm = nn.LSTM(input_dim, (input_dim*2)//3, num_layers=2,
                            batch_first=True, bidirectional=bidirectional)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

        self.cell_and_hidden = self.init_cell_and_hidden()
        return 

    def init_cell_and_hidden(self):
        return (Variable(torch.FloatTensor(1, 1, self.hidden_dim).normal_()),
                Variable(torch.FloatTensor(1, 1, self.hidden_dim).normal_()))

    def forward(self, inpt):
        chord_seq = inpt.view(inpt.size()[0], np.prod(inpt.size()[1:]))
        lstm_out, self.cell_and_hidden = self.lstm(chord_seq, self.cell_and_hidden)
        out = self.sigmoid(self.fc_out(lstm_out))
        return out


# class PitchNet(nn.Module):
#     def __init__(self, chord_dim, embedding_dim, **kwargs):
#         super(DeepJazzClone, self).__init__(**kwargs)
#         self.fc_chord_input = nn.Linear(chord_dim, embedding_dim)
#         self.pitch_class_embeddings = nn.Embedding(NUM_PITCH_CLASSES, embedding_dim)
#         self.octave_embeddings = nn.Embedding(NUM_DURATION_CLASSES, embedding_dim)
#         # self.lstm = nn.LSTM(
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



