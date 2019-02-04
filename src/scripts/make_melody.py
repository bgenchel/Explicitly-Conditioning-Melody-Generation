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

PITCH_NUM_OFFSET = 21 # A0

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from models.no_cond.model_classes import PitchLSTM as NoCondPitch, DurationLSTM as NoCondDur
from models.inter_cond.model_classes import PitchLSTM as InterCondPitch, DurationLSTM as InterCondDur
from models.chord_cond.model_classes import PitchLSTM as ChordCondPitch, DurationLSTM as ChordCondDur
from models.barpos_cond.model_classes import PitchLSTM as BarPosCondPitch, DurationLSTM as BarPosCondDur
from models.chord_inter_cond.model_classes import (
        PitchLSTM as ChordInterCondPitch, 
        DurationLSTM as ChordInterCondDur)
from models.chord_barpos_cond.model_classes import (
        PitchLSTM as ChordBarPosCondPitch, 
        DurationLSTM as ChordBarPosCondDur)
from models.inter_barpos_cond.model_classes import (
        PitchLSTM as InterBarPosCondPitch, 
        DurationLSTM as InterBarPosCondDur)
from models.chord_inter_barpos_cond.model_classes import (
        PitchLSTM as ChordInterBarPosCondPitch, 
        DurationLSTM as ChordInterBarPosCondDur)

import utils.constants as const
from utils.reverse_pianoroll import piano_roll_to_pretty_midi

INIT_TO_MODEL = {'nc': {'name': 'no_cond',
                        'pitch_model': NoCondPitch,
                        'duration_model': NoCondDur}, 
                 'ic': {'name': 'inter_cond',
                        'pitch_model': InterCondPitch,
                        'duration_model': InterCondDur},
                 'cc': {'name': 'chord_cond', 
                        'pitch_model': ChordCondPitch,
                        'duration_model': ChordCondDur},
                 'bc': {'name': 'barpos_cond', 
                        'pitch_model': BarPosCondPitch,
                        'duration_model': BarPosCondDur},
                 'cic': {'name': 'chord_inter_cond',
                        'pitch_model': ChordInterCondPitch,
                        'duration_model': ChordInterCondDur},
                 'cbc': {'name': 'chord_barpos_cond',
                        'pitch_model': ChordBarPosCondPitch,
                        'duration_model': ChordBarPosCondDur},
                 'ibc': {'name': 'inter_barpos_cond',
                        'pitch_model': InterBarPosCondPitch,
                        'duration_model': InterBarPosCondDur},
                 'cibc': {'name': 'chord_inter_barpos_cond',
                        'pitch_model': ChordInterBarPosCondPitch,
                        'duration_model': ChordInterBarPosCondDur}}

CHORD_OFFSET = 48 # chords will be in octave 3
# BUFF_LEN = 32

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--title', default="generated", type=str,
                    help="what to name the output")
parser.add_argument('-m', '--model', type=str, choices=("nc", "ic", "cc", "bc", "cic", "cbc", "ibc", "cibc"), 
                    help="which model to use for generation.\n \
                          \t\tnc - no_cond, ic - inter_cond, cc - chord_inter_cond, bc - barpos_cond \n \
                          \t\tcic - chord_inter_barpos_cond, cbc - chord_barpos_Cond, ibc - inter_barpos_cond \n \
                          \t\tcibc - chord_inter_barpos_cond.")
parser.add_argument('-pn', '--pitch_run_name', type=str,
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str,
                    help="select which dur run to use")
parser.add_argument('-ss', '--gen_song', type=str, default=None,
                    help="which song to generate over.")
# parser.add_argument('-sm', '--seed_measures', type=int, default=1,
#                     help="number of measures to use as seeds to the network")
parser.add_argument('-sl', '--seed_len', type=int, default=3,
                    help="number of events to use for the seed")
parser.add_argument('-nr', '--num_repeats', type=int, default=1,
                    help="generate over the chords (nr) times.")
args = parser.parse_args()


def convert_melody_to_piano_roll_mat(pitches, dur_nums):
    dur_ticks = [const.DUR_TICKS_MAP[const.REV_DURATIONS_MAP[dur]] for dur in dur_nums]
    onsets = np.array([np.sum(dur_ticks[:i]) for i in range(len(dur_ticks))])
    total_ticks = sum(dur_ticks)
    output_mat = np.zeros([128, int(total_ticks)])
    for i in range(len(pitches) - 1):
        if pitches[i - 1] == const.NOTES_MAP['rest']:
            continue
        else:
            # include the -1 for now because stuff is out of key
            output_mat[int(pitches[i - 1]) + PITCH_NUM_OFFSET, int(onsets[i]):int(onsets[i+1])] = 1.0
    output_mat[int(pitches[-1] - 1) + PITCH_NUM_OFFSET, int(onsets[-1]):] = 1.0
    return output_mat


def convert_chords_to_piano_roll_mat(song_structure):
    output_mat = np.zeros([128, len(song_structure)*const.DUR_TICKS_MAP['whole']])
    for i, measure in enumerate(song_structure):
        ticks = i*const.DUR_TICKS_MAP['whole']
        for j, group in enumerate(measure):
            chord_vec, begin, end = group
            chord_block = np.array(chord_vec).reshape((len(chord_vec), 1)).repeat(end - begin, axis=1)
            output_mat[CHORD_OFFSET:CHORD_OFFSET + len(chord_vec), ticks + begin:ticks + end] = chord_block
    return output_mat


root_dir = str(Path(op.abspath(__file__)).parents[2])
model_dir = op.join(root_dir, 'src', 'models', INIT_TO_MODEL[args.model]['name'])

pitch_dir = op.join(model_dir, 'runs', 'pitch', args.pitch_run_name)
dur_dir = op.join(model_dir, 'runs', 'duration', args.dur_run_name)

pitch_model_inputs = json.load(open(op.join(pitch_dir, 'model_inputs.json'), 'r'))
pitch_model_inputs['batch_size'] = 1
pitch_model_inputs['dropout'] = 0
pitch_model_inputs['no_cuda'] = True
pitch_model_state = torch.load(op.join(pitch_dir, 'model_state.pt'), map_location='cpu')

PitchModel = INIT_TO_MODEL[args.model]['pitch_model'](**pitch_model_inputs)
PitchModel.load_state_dict(pitch_model_state)
PitchModel.eval()

dur_model_inputs = json.load(open(op.join(dur_dir, 'model_inputs.json'), 'r'))
dur_model_inputs['batch_size'] = 1
dur_model_inputs['dropout'] = 0
dur_model_inputs['no_cuda'] = True
dur_model_state = torch.load(op.join(dur_dir, 'model_state.pt'), map_location='cpu')

DurModel = INIT_TO_MODEL[args.model]['duration_model'](**dur_model_inputs)
DurModel.load_state_dict(dur_model_state)
DurModel.eval()

PitchModel.init_hidden_and_cell(1)
DurModel.init_hidden_and_cell(1)

data_dir = op.join(root_dir, 'data', 'processed', 'songs')
if args.gen_song is None:
    songs = os.listdir(data_dir)
    gen_song = pickle.load(open(op.join(data_dir, random.choice(songs)), 'rb'))
else:
    gen_song = pickle.load(open(op.join(data_dir, args.gen_song), 'rb'))

seed_data = {const.PITCH_KEY: [], const.DUR_KEY: [], const.CHORD_KEY: [], const.BARPOS_KEY: []}
for i, measure in enumerate(gen_song["measures"]):
    for j, group in enumerate(measure["groups"]):
        assert len(group[const.PITCH_KEY]) == len(group[const.DUR_KEY]) == len(group[const.BARPOS_KEY])
        seed_data[const.PITCH_KEY].extend(group[const.PITCH_KEY])
        seed_data[const.DUR_KEY].extend(group[const.DUR_KEY])
        seed_data[const.BARPOS_KEY].extend(group[const.BARPOS_KEY])

        chord_vec = group["harmony"]["root"] + group["harmony"]["pitch_classes"]
        # right now each element is actual just pointers to one list, which is really bad
        # however, this problem will be resolved when converted to tensor/np array
        seed_data[const.CHORD_KEY].extend([chord_vec]*len(group[const.PITCH_KEY]))
seed_data = {k: np.array(v[:args.seed_len]) for k, v in seed_data.items()}
# shuffle the chords and harmonies for a somewhat random seed with accurate timing info
seed_data_groups = list(zip(seed_data[const.PITCH_KEY], seed_data[const.CHORD_KEY]))
random.shuffle(seed_data_groups)
seed_data[const.PITCH_KEY], seed_data[const.CHORD_KEY] = zip(*seed_data_groups)

# these two will allow the generation of a new melody by keeping track of within measure
# chord and bar positions.
song_structure = []
for i, measure in enumerate(gen_song["measures"]):
    measure_groups = []
    for j, group in enumerate(measure["groups"]):
        # my worrying assumption about this is that there will be places where a chord floats above, but
        # no notes are actually present. Also, what about tied notes that tie to a place that's longer
        # not just the first beat of the first bar? Just gonna not worry about that for now.
        chord_vec = group["harmony"]["root"] + group["harmony"]["pitch_classes"]
        # print('group[const.BARPOS_KEY]: {}'.format(group[const.BARPOS_KEY]))
        # print('group[const.PITCH_KEY]: {}'.format(group[const.PITCH_KEY]))
        # print('group[const.DUR_KEY]: {}'.format(group[const.DUR_KEY]))
        begin = group[const.BARPOS_KEY][0]
        end = group[const.BARPOS_KEY][-1] + const.DUR_TICKS_MAP[const.REV_DURATIONS_MAP[group[const.DUR_KEY][-1]]]
       
        measure_groups.append((chord_vec, begin, end))
    song_structure.append(measure_groups)

pitch_inpt = torch.LongTensor(seed_data[const.PITCH_KEY]).view(1, -1)
dur_inpt = torch.LongTensor(seed_data[const.DUR_KEY]).view(1, -1)
barpos_inpt = torch.LongTensor(seed_data[const.BARPOS_KEY]).view(1, -1)
harmony_inpt = torch.FloatTensor(seed_data[const.CHORD_KEY]).view(1, -1, const.CHORD_DIM)

pitch_seq = []
dur_seq = []
barpos_seq = []
harmony_seq = []
for _ in range(args.num_repeats):
    for i, measure in enumerate(song_structure):
        curr_barpos = 0
        for j, group in enumerate(measure):
            chord, begin, end = group
            while curr_barpos < end:
                if args.model == "nc":
                    pitch_out = PitchModel(pitch_inpt)
                    dur_out = DurModel(dur_inpt)
                elif args.model == "ic":
                    pitch_out = PitchModel((pitch_inpt, dur_inpt))
                    dur_out = DurModel((dur_inpt, pitch_inpt))
                elif args.model == "cc":
                    pitch_out = PitchModel((pitch_inpt, harmony_inpt))
                    dur_out = DurModel((dur_inpt, harmony_inpt))
                elif args.model == "bc":
                    pitch_out = PitchModel((pitch_inpt, barpos_inpt))
                    dur_out = DurModel((dur_inpt, barpos_inpt))
                elif args.model == "cic":
                    pitch_out = PitchModel((pitch_inpt, dur_inpt, harmony_inpt))
                    dur_out = DurModel((dur_inpt, pitch_inpt, harmony_inpt))
                elif args.model == "cbc":
                    pitch_out = PitchModel((pitch_inpt, barpos_inpt, harmony_inpt))
                    dur_out = DurModel((dur_inpt, barpos_inpt, harmony_inpt))
                elif args.model == "ibc":
                    pitch_out = PitchModel((pitch_inpt, dur_inpt, barpos_inpt))
                    dur_out = DurModel((dur_inpt, pitch_inpt, barpos_inpt))
                elif args.model == "cibc":
                    pitch_out = PitchModel((pitch_inpt, dur_inpt, barpos_inpt, harmony_inpt))
                    dur_out = DurModel((dur_inpt, pitch_inpt, barpos_inpt, harmony_inpt))

                pitch_seq.append(int(torch.exp(pitch_out.data[:, -1, :]).multinomial(1)))
                dur_seq.append(int(torch.exp(dur_out.data[:, -1, :]).multinomial(1)))
                barpos_seq.append(curr_barpos)
                harmony_seq.append(chord)

                curr_barpos += const.DUR_TICKS_MAP[const.REV_DURATIONS_MAP[dur_seq[-1]]]
                pitch_inpt = torch.LongTensor([pitch_seq[-1]]).view(1, -1)
                dur_inpt = torch.LongTensor([dur_seq[-1]]).view(1, -1)
                barpos_inpt = torch.LongTensor([barpos_seq[-1]]).view(1, -1)
                harmony_inpt = torch.FloatTensor(harmony_seq[-1]).view(1, -1, const.CHORD_DIM)
        
melody_pr_mat = convert_melody_to_piano_roll_mat(pitch_seq, dur_seq)
chords_pr_mat = convert_chords_to_piano_roll_mat(song_structure)
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
