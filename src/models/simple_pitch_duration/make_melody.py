import argparse
import json
import numpy
import os
import os.path as op
import torch
from torch.autograd import Variable
from models import PitchLSTM, DurationLSTM

parser = argparse.ArgumentParser()
parser.add_argument('-pn', '--pitch_run_name', type=str,
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str,
                    help="select which dur run to use")
parser.add_argument('-k', '--keep', action='store_true',
                    help="save information about this run")
args = parser.parse_args()

pitch_dir = 
pitch_net = 
