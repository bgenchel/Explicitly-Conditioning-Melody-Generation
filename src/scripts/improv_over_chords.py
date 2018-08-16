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

sys.path.append(Path(op.abspath(__file__)).parents[1])
from models.simple_pitch_duration.model_classes import BaselineLSTM
from models.chord_conditioning.model_classes import ChordCondLSTM
from models.chord_and_inter_conditioning.model_classes import ChordandInterConditionedLSTM
from processing.parsing.harmony import Harmony
from utils.reverse_pianoroll import piano_roll_to_pretty_midi
from utils.constants import NOTES_MAP, DURATIONS_MAP

PITCH_RUN = "2018-08-07_07:58:30_CP"
DUR_RUN = "2018-08-07_08:11:52_CP"
MODEL = "chord_and_inter_conditioning"
CTFILE = "iltur.json"

NAME_TO_MODEL = {'simple_pitch_duration': BaselineLSTM, 
                 'chord_conditioning': ChordCondLSTM,
                 'chord_and_inter_conditioning': ChordandInterConditionedLSTM}

PITCH_KEY = "pitch_numbers"
DUR_KEY = "duration_tags"
FILLER = {PITCH_KEY: NOTES_MAP['rest'], DUR_KEY: DURATIONS_MAP['none']}

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

root_dir = str(Path(op.abspath(__file__)).parents[2])
model_dir = op.join(root_dir, 'src', 'models', MODEL)
pitch_dir = op.join(model_dir, 'runs', 'pitch', PITCH_RUN)
dur_dir = op.join(model_dir, 'runs', 'duration', DUR_RUN)

pitch_model_inputs = json.load(open(op.join(pitch_dir, 'model_inputs.json'), 'r'))
pitch_model_inputs['batch_size'] = 1
pitch_model_inputs['dropout'] = 0
pitch_model_inputs['no_cuda'] = True
pitch_model_state = torch.load(op.join(pitch_dir, 'model_state.pt'), map_location='cpu')
pitch_net = NAME_TO_MODEL[MODEL](**pitch_model_inputs)
pitch_net.load_state_dict(pitch_model_state)
pitch_net.eval()

dur_model_inputs = json.load(open(op.join(dur_dir, 'model_inputs.json'), 'r'))
dur_model_inputs['batch_size'] = 1
dur_model_inputs['dropout'] = 0
dur_model_inputs['no_cuda'] = True
dur_model_state = torch.load(op.join(dur_dir, 'model_state.pt'), map_location='cpu')
dur_net = NAME_TO_MODEL[MODEL](**dur_model_inputs)
dur_net.load_state_dict(dur_model_state)
dur_net.eval()

data_dir = op.join(root_dir, "data", "chord_tick_files")
chord_data = json.load(open(op.join(data_dir, CTFILE)))
assert isinstance(chord_data, list)
chord_data = [{"chord: :q



