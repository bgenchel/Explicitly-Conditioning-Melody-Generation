import argparse
import json
import numpy as np
import os
import os.path as op
import pickle
import random
import torch
from torch.autograd import Variable
from models import PitchLSTM, DurationLSTM
from pathlib import Path
from reverse_pianoroll import piano_roll_to_pretty_midi

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default="melody.mid", type=str,
                    help="what to name the output")
parser.add_argument('-pn', '--pitch_run_name', type=str,
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str,
                    help="select which dur run to use")
parser.add_argument('-l', '--seq_len', default=40, type=int,
                    help="how many notes/durs to generate")
args = parser.parse_args()

# just use indices instead of making a dict with number keys
NUM_TO_TAG = ['whole', 'half', 'quarter', 'eighth', '16th', 'whole-triplet', 
              'half-triplet', 'quarter-triplet', 'eighth-triplet', '16th-triplet', 
              'whole-dot', 'half-dot', 'quarter-dot', 'eighth-dot', '16th-dot', 
              '32nd', '32nd-triplet', '32nd-dot']

# assumes 96 ticks to a bar
TAG_TO_TICKS = {'whole': 96, 'half': 48, 'quarter': 24, 'eighth': 12, '16th': 6, 
                 'whole-triplet': 64, 'half-triplet': 32, 'quarter-triplet': 16, 
                 'eighth-triplet': 8, '16th-triplet': 4, 'whole-dot': 144, 'half-dot': 72, 
                 'quarter-dot': 36, 'eighth-dot': 18, '16th-dot': 9, '32nd': 3, 
                 '32nd-triplet': 2, '32nd-dot': 5, 'other': -1}

def convert_to_piano_roll_mat(pitches, dur_nums):
    print(dur_nums)
    dur_ticks = [TAG_TO_TICKS[NUM_TO_TAG[dur]] for dur in dur_nums]
    onsets = np.array([np.sum(dur_ticks[:i]) for i in range(len(dur_ticks))])
    total_ticks = sum(dur_ticks)
    output_mat = np.zeros([128, int(total_ticks)])
    # pdb.set_trace()
    for i in range(len(pitches) - 1):
        if pitches[i] == 0:
            continue
        else:
            output_mat[int(pitches[i]), int(onsets[i]):int(onsets[i+1])] = 1.0
    # pdb.set_trace()
    output_mat[int(pitches[-1]), int(onsets[-1]):] = 1.0
    return output_mat

pitch_dir = op.join(os.getcwd(), 'runs', 'pitches', args.pitch_run_name)
dur_dir = op.join(os.getcwd(), 'runs', 'durations', args.dur_run_name)

pitch_model_inputs = json.load(open(op.join(pitch_dir, 'model_inputs.json'), 'r'))
pitch_model_inputs['batch_size'] = 1
pitch_model_state = torch.load(op.join(pitch_dir, 'model_state.pt'))
pitch_net = PitchLSTM(**pitch_model_inputs)
pitch_net.load_state_dict(pitch_model_state)

dur_model_inputs = json.load(open(op.join(dur_dir, 'model_inputs.json'), 'r'))
dur_model_inputs['batch_size'] = 1
dur_model_state = torch.load(op.join(dur_dir, 'model_state.pt'))
dur_net = DurationLSTM(**dur_model_inputs)
dur_net.load_state_dict(dur_model_state)

root_dir = str(Path(op.abspath(__file__)).parents[3])
data_dir = op.join(root_dir, 'data', 'processed', 'songs')
songs = os.listdir(data_dir)
seed_song = pickle.load(open(op.join(data_dir, random.choice(songs)), 'rb'))
seed_song_pitches = []
seed_song_durs = []
for measure in seed_song['measures']:
    seed_song_pitches.extend(measure['pitch_numbers'])
    seed_song_durs.extend(measure['duration_tags'])

seed_pitches = [random.choice(seed_song_pitches) for _ in range(2)]
pitch_inpt = Variable(torch.LongTensor(seed_pitches)).view(1, -1)
seed_durs = [random.choice(seed_song_durs) for _ in range(2)]
dur_inpt = Variable(torch.LongTensor(seed_durs)).view(1, -1)

pitch_seq = seed_pitches
dur_seq = seed_durs
for _ in range(args.seq_len):
    pitch_out = pitch_net(pitch_inpt)
    pitch_seq.append(np.argmax(pitch_out.data[:, -1, :]))
    pitch_inpt = Variable(torch.LongTensor(pitch_seq[-2:]).view(1, -1))

    dur_out = dur_net(dur_inpt)
    dur_seq.append(np.argmax(dur_out.data[:, -1, :]))
    dur_inpt = Variable(torch.LongTensor(dur_seq[-2:]).view(1, -1))

pr_mat = convert_to_piano_roll_mat(pitch_seq, dur_seq)
pm = piano_roll_to_pretty_midi(pr_mat, fs=30)
outpath = op.join(os.getcwd(), 'midi', '%s.mid' % args.title)
print('Writing output midi file %s ...' % outpath)
pm.write(outpath)
