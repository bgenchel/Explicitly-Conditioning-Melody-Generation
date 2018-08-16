import argparse
import json
import numpy as np
import os
import os.path as op
import pickle
import random
import sys
import torch
from torch.autograd import Variable
from pathlib import Path

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from models.simple_pitch_duration.model_classes import BaselineLSTM
from models.chord_conditioning.model_classes import ChordCondLSTM
from models.chord_and_inter_conditioning.model_classes import ChordandInterConditionedLSTM
from utils.reverse_pianoroll import piano_roll_to_pretty_midi
from utils.constants import NOTES_MAP, DURATIONS_MAP

##### MONKEY PATCH #####
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
########## 

REST_NUM = 127

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

NUM_TO_MODEL = {1: {'name': 'simple_pitch_duration',
                    'model': BaselineLSTM}, 
                2: {'name': 'chord_conditioning', 
                    'model': ChordCondLSTM},
                3: {'name': 'chord_and_inter_conditioning',
                    'model': ChordandInterConditionedLSTM}}

PITCH_KEY = "pitch_numbers"
DUR_KEY = "duration_tags"
FILLER = {PITCH_KEY: NOTES_MAP['rest'], DUR_KEY: DURATIONS_MAP['none']}

CHORD_DIM = 12
CHORD_OFFSET = 60 # chords will be in octave 4
BUFF_LEN = 16

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default="generated", type=str,
                    help="what to name the output")
parser.add_argument('-m', '--model', type=int,
                    choices=(1, 2, 3), help="which model to use for generation")
parser.add_argument('-pn', '--pitch_run_name', type=str,
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str,
                    help="select which dur run to use")
parser.add_argument('-sm', '--seed_measures', type=int, default=1,
                    help="number of measures to use as seeds to the network")
parser.add_argument('-ss', '--seed_song', type=str, default=None,
                    help="which song to generate over.")
parser.add_argument('-nr', '--num_repeats', type=int, default=1,
                    help="generate over the chords (nr) times.")
args = parser.parse_args()

def convert_melody_to_piano_roll_mat(pitches, dur_nums):
    dur_ticks = [TAG_TO_TICKS[NUM_TO_TAG[dur]] for dur in dur_nums]
    onsets = np.array([np.sum(dur_ticks[:i]) for i in range(len(dur_ticks))])
    total_ticks = sum(dur_ticks)
    output_mat = np.zeros([128, int(total_ticks)])
    for i in range(len(pitches) - 1):
        if pitches[i - 1] == REST_NUM:
            continue
        else:
            # include the -1 for now because stuff is out of key
            output_mat[int(pitches[i - 1]), int(onsets[i]):int(onsets[i+1])] = 1.0
    output_mat[int(pitches[-1] - 1), int(onsets[-1]):] = 1.0
    return output_mat

def convert_chords_to_piano_roll_mat(note_chords, dur_nums):
    dur_ticks = [TAG_TO_TICKS[NUM_TO_TAG[dur]] for dur in dur_nums]
    onsets = np.array([np.sum(dur_ticks[:i]) for i in range(len(dur_ticks))])
    total_ticks = sum(dur_ticks)
    output_mat = np.zeros([128, int(total_ticks)])
    for i in range(len(onsets) - 1):
        for j in range(len(note_chords[i])):
            if note_chords[i][j] == 1:
                chord_tone = j + CHORD_OFFSET
                output_mat[chord_tone, int(onsets[i]):(int(onsets[i+1]))] = 1.0
    for j in range(len(note_chords[-1])):
        if j == 1:
            chord_tone = j + CHORD_OFFSET
            output_mat[chord_tone, int(onsets[-1]):] = 1.0
    return output_mat

root_dir = str(Path(op.abspath(__file__)).parents[2])
model_dir = op.join(root_dir, 'src', 'models', NUM_TO_MODEL[args.model]['name'])

pitch_dir = op.join(model_dir, 'runs', 'pitch', args.pitch_run_name)
dur_dir = op.join(model_dir, 'runs', 'duration', args.dur_run_name)

pitch_model_inputs = json.load(open(op.join(pitch_dir, 'model_inputs.json'), 'r'))
pitch_model_inputs['batch_size'] = 1
pitch_model_inputs['dropout'] = 0
pitch_model_inputs['no_cuda'] = True
pitch_model_state = torch.load(op.join(pitch_dir, 'model_state.pt'), map_location='cpu')
pitch_net = NUM_TO_MODEL[args.model]['model'](**pitch_model_inputs)
pitch_net.load_state_dict(pitch_model_state)
pitch_net.eval()

dur_model_inputs = json.load(open(op.join(dur_dir, 'model_inputs.json'), 'r'))
dur_model_inputs['batch_size'] = 1
dur_model_inputs['dropout'] = 0
dur_model_inputs['no_cuda'] = True
dur_model_state = torch.load(op.join(dur_dir, 'model_state.pt'), map_location='cpu')
dur_net = NUM_TO_MODEL[args.model]['model'](**dur_model_inputs)
dur_net.load_state_dict(dur_model_state)
dur_net.eval()

data_dir = op.join(root_dir, 'data', 'processed', 'songs')
if args.seed_song is None:
    songs = os.listdir(data_dir)
    seed_song = pickle.load(open(op.join(data_dir, random.choice(songs)), 'rb'))
else:
    seed_song = pickle.load(open(op.join(data_dir, args.seed_song), 'rb'))

seed_song_chords = []
for measure in seed_song['measures']:
    seed_song_chords.append(measure['harmonies'])

seed_pitches = []
seed_durs = []
seed_note_chords = []
for measure in seed_song['measures'][:args.seed_measures]:
    measure_pitches = measure[PITCH_KEY]
    measure_durs = measure[DUR_KEY]
    measure_note_chords = []
    harmony_index = 0
    for j in range(len(measure_pitches)):
        measure_note_chords.append(measure['harmonies'][harmony_index])
        if (j == measure['half_index']) and (len(measure['harmonies']) > 1):
            harmony_index += 1
    seed_pitches.extend(measure_pitches)
    seed_durs.extend(measure_durs)
    seed_note_chords.extend(measure_note_chords)

seed_groups = list(zip(seed_pitches, seed_durs, seed_note_chords))
random.shuffle(seed_groups)
seed_pitches, seed_durs, seed_note_chords = zip(*seed_groups)

pitches_buff = np.array([FILLER[PITCH_KEY]]*(BUFF_LEN))
pitches_buff[-len(seed_pitches):] = seed_pitches
durs_buff = np.array([FILLER[DUR_KEY]]*(BUFF_LEN))
durs_buff[-len(seed_durs):] = seed_durs
note_chords_buff = np.array([(lambda: [0]*CHORD_DIM)() for _ in range(BUFF_LEN)])
note_chords_buff[-len(seed_note_chords):] = seed_note_chords

chord_inpt = Variable(torch.FloatTensor(note_chords_buff)).view(1, -1, CHORD_DIM)
pitch_inpt = Variable(torch.LongTensor(pitches_buff)).view(1, -1)
dur_inpt = Variable(torch.LongTensor(durs_buff)).view(1, -1)

note_chord_seq = []
pitch_seq = []
dur_seq = []
for _ in range(args.num_repeats):
    for i, measure_chords in enumerate(seed_song_chords):
        tick_lim = TAG_TO_TICKS["whole"]
        if len(measure_chords) > 1:
            tick_lim = TAG_TO_TICKS["half"]
        
        for chord in measure_chords:
            total_ticks = 0
            while total_ticks < tick_lim:
                if args.model == 3:
                    pitch_out = pitch_net(chord_inpt, dur_inpt, pitch_inpt)
                    dur_out = dur_net(chord_inpt, pitch_inpt, dur_inpt)
                elif args.model == 2:
                    pitch_out = pitch_net(chord_inpt, pitch_inpt)
                    dur_out = dur_net(chord_inpt, dur_inpt)
                elif args.model == 1:
                    pitch_out = pitch_net(pitch_inpt)
                    dur_out = dur_net(dur_inpt)

                new_pitch = np.argmax(pitch_out.data[:, -1, :])
                new_dur = np.argmax(dur_out.data[:, -1, :])
                total_ticks += TAG_TO_TICKS[NUM_TO_TAG[new_dur]]

                pitches_buff[0] = new_pitch
                pitches_buff = np.roll(pitches_buff, -1)
                pitch_inpt = Variable(torch.LongTensor(pitches_buff).view(1, -1))
                durs_buff[0] = new_dur
                durs_buff = np.roll(durs_buff, -1)
                dur_inpt = Variable(torch.LongTensor(durs_buff).view(1, -1))
                note_chords_buff[0] = chord
                note_chords_buff = np.roll(note_chords_buff, -1)
                chord_inpt = Variable(torch.FloatTensor(note_chords_buff).view(1, -1, CHORD_DIM))

                pitch_seq.append(new_pitch)
                dur_seq.append(new_dur)
                note_chord_seq.append(chord)

melody_pr_mat = convert_melody_to_piano_roll_mat(pitch_seq, dur_seq)
chords_pr_mat = convert_chords_to_piano_roll_mat(note_chord_seq, dur_seq)
melody_pm = piano_roll_to_pretty_midi(melody_pr_mat, fs=30)
chords_pm = piano_roll_to_pretty_midi(chords_pr_mat, fs=30)
outdir = op.join(model_dir, 'midi', args.title)
if not op.exists(outdir):
    os.makedirs(outdir)
melody_path = op.join(outdir, '%s_melody.mid' % args.title)
chords_path = op.join(outdir, '%s_chords.mid' % args.title)
print('Writing melody midi file %s ...' % melody_path)
melody_pm.write(melody_path)
print('Writing chords midi file %s ...' % chords_path)
chords_pm.write(chords_path)
