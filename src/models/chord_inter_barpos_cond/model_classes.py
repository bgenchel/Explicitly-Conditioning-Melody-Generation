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

class ChordInterBarPosCondLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, cond_vocab_size, cond_embed_dim, output_dim, hidden_dim=128,
            seq_len=32, batch_size=64, dropout=0.5, batch_norm=True, no_cuda=False, **kwargs):
        super().__init__(**kwargs) 
        self.hidden_dim = hidden_dim
        self.num_layers = const.NUM_RNN_LAYERS
        self.batch_norm = batch_norm
        self.no_cuda = no_cuda

        self.chord_fc1 = nn.Linear(const.CHORD_DIM, const.CHORD_EMBED_DIM)
        self.chord_bn = nn.BatchNorm1d(seq_len)
        self.chord_fc2 = nn.Linear(const.CHORD_EMBED_DIM, const.CHORD_EMBED_DIM)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cond_embedding = nn.Embedding(cond_vocab_size, cond_embed_dim)
        self.barpos_embedding = nn.Embedding(const.BARPOS_DIM, const.BARPOS_EMBED_DIM)

        encoder_input_dim = embed_dim + cond_embed_dim + const.CHORD_EMBED_DIM + const.BARPOS_EMBED_DIM
        self.encoder = nn.Linear(encoder_input_dim, hidden_dim)
        self.encoder_bn = nn.BatchNorm1d(seq_len)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=dropout)

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
        return 

    def init_hidden_and_cell(self, batch_size):
        hidden = Variable(torch.zeros([self.num_layers, batch_size, self.hidden_dim]))
        cell = Variable(torch.zeros([self.num_layers, batch_size, self.hidden_dim]))
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

    def forward(self, data):
        x, conds, barpos, chords = data

        chord_embeds = self.chord_fc1(chords)
        if self.batch_norm:
            chord_embeds = self.chord_bn(chord_embeds)
        chord_embeds = self.chord_fc2(F.relu(chord_embeds))

        x_embeds = self.embedding(x)
        cond_embeds = self.cond_embedding(conds)
        barpos_embeds = self.barpos_embedding(barpos)

        # Concatenating along 3rd dimension
        encoding = self.encoder(torch.cat([x_embeds, cond_embeds, barpos_embeds, chord_embeds], 2)) 
        if self.batch_norm:
            encoding = self.encode_bn(encoding)
        encoding = F.relu(encoding)

        lstm_out, self.hidden_and_cell = self.lstm(encoding, self.hidden_and_cell)
        decoding = self.decode1(lstm_out)
        if self.batch_norm:
            decoding = self.decode_bn(decoding)
        decoding = self.decode2(F.relu(decoding))

        output = self.softmax(decoding)
        return output

class PitchLSTM(ChordInterBarPosCondLSTM):
    def __init__(self, **kwargs):
        super().__init__(vocab_size=const.PITCH_DIM, embed_dim=const.PITCH_EMBED_DIM,
                         cond_vocab_size=const.DUR_DIM, cond_embed_dim=const.DUR_EMBED_DIM,
                         output_dim=const.PITCH_DIM, **kwargs)

    def data_assembler(self, data_dict):
        x = data_dict[const.PITCH_KEY]
        cond = data_dict[const.DUR_KEY]
        barpos = data_dict[const.BARPOS_KEY]
        harmony = data_dict[const.CHORD_KEY]
        if torch.cuda.is_available() and (not self.no_cuda):
            x = x.cuda()
            cond = cond.cuda()
            barpos = barpos.cuda()
            harmony = harmony.cuda()
        return (x, cond, barpos, harmony)

    def target_assembler(self, target_dict):
        target = target_dict[const.PITCH_KEY]
        if torch.cuda.is_available() and (not self.no_cuda):
            target = target.cuda()
        return target

class DurationLSTM(ChordInterBarPosCondLSTM):
    def __init__(self, **kwargs):
        super().__init__(vocab_size=const.DUR_DIM, embed_dim=const.DUR_EMBED_DIM,
                         cond_vocab_size=const.PITCH_DIM, cond_embed_dim=const.PITCH_EMBED_DIM,
                         output_dim=const.DUR_DIM, **kwargs)

    def data_assembler(self, data_dict):
        x = data_dict[const.DUR_KEY]
        cond = data_dict[const.PITCH_KEY]
        barpos = data_dict[const.BARPOS_KEY]
        harmony = data_dict[const.CHORD_KEY]
        if torch.cuda.is_available() and (not self.no_cuda):
            x = x.cuda()
            cond = cond.cuda()
            barpos = barpos.cuda()
            harmony = harmony.cuda()
        return (x, cond, barpos, harmony)

    def target_assembler(self, target_dict):
        target = target_dict[const.DUR_KEY]
        if torch.cuda.is_available() and (not self.no_cuda):
            target = target.cuda()
        return target
