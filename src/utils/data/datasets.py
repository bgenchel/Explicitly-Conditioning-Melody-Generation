import copy
import numpy as np
import os
import os.path as op
import pickle
import sys
import torch
from pathlib import Path
from torch.utils.data.dataset import Dataset

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
import constants as const

PITCH_KEY = "pitch_numbers"
const.DUR_KEY = "duration_tags"
const.CHORD_KEY = "harmony"
const.POS_KEY = "bar_positions"

DEFAULT_DATA_PATH = op.join(Path(op.abspath(__file__)).parents[3], 'data', 'processed', 'songs')

class BebopPitchDurDataset(Dataset):
    """
    This defines the loading and organizing of parsed MusicXML data into a dataset of note pitch and note duration
    values.
    """

    def __init__(self, load_dir=DEFAULT_DATA_PATH, seq_len=32, train_type="next_step"):
        """
        Loads the MIDI tick information, groups into sequences based on measures.
        :param load_dir: location of parsed MusicXML data
        :param seq_len: how long a single training sequence is
        :param data: this dataset will be used to get either the pitch tokens or the duration tokens from the data, this
            parameter specifies which of the two to get. Valid options are "pitch" and "duration".
        :param train_type: this param specifies the format of the target and thus the way in which the network will be
            trained. Valid arguments are "full_sequence" (which specifies that the target will be a sequence the same
            length as the input, but will be taken to be shifted forward one step in time), and "next_step" (in which
            the targets are only the next step after the input, and only the last output of the network is considered).
        :param chord_cond: "chord conditioning" - include harmony, in the form of a 24 length binary vector. The first 12 bits specify the root
            note in one hot format, while the latter 12 specify the included pitch classes.
        :param inter_cond: "interconditioning" - if data is "pitch", also give duration tokens; if data is "duration"
            also give pitch numbers. 
        :param bar_pos_cond: "bar position conditioning" - give a bar position number which specifies the position of 
            each note in a bar. 
        """

        if not op.exists(load_dir):
            raise Exception("Data directory does not exist.")

        self.seq_len = seq_len
        self.train_type = train_type
        
        self.sequences = self._create_data_dict()
        self.targets = self._create_data_dict()
        print('Loading files from %s ...' % load_dir)
        for fname in os.listdir(load_dir):
            if op.splitext(fname)[1] != ".pkl":
                print("Skipping %s..." % fname)
                continue

            song = pickle.load(open(op.join(load_dir, fname), "rb"))

            if song["metadata"]["time_signature"] != "4/4":
                print("Skipping %s because it isn't in 4/4." % fname)

            full_sequence = self._create_data_dict()
            for i, measure in enumerate(song["measures"]):
                for j, group in enumerate(measure["groups"]):
                    assert len(group[const.PITCH_KEY]) == len(group[const.DUR_KEY]) == len(group[const.POS_KEY])
                    full_sequence[const.PITCH_KEY].extend(group[const.PITCH_KEY])
                    full_sequence[const.DUR_KEY].extend(group[const.DUR_KEY])
                    full_sequence[const.POS_KEY].extend(group[const.POS_KEY])

                    chord_vec = group["harmony"]["root"] + group["harmony"]["pitch_classes"]
                    # right now each element is actual just pointers to one list, which is really bad
                    # however, this problem will be resolved when converted to tensor/np array
                    for _ in range(len(group[const.PITCH_KEY])): 
                        full_sequence[const.CHORD_KEY].append(chord_vec)

            full_sequence = {k: np.array(v) for k, v in full_sequence.items()}
            for k, full_seq in full_sequence.items():
                seqs, targets = self._get_seqs_and_targets(full_seq)
                self.sequences[k].extend(seqs)
                self.targets[k].extend(targets)

    def _create_data_dict(self):
        return {const.PITCH_KEY: [], const.DUR_KEY: [], const.CHORD_KEY: [], const.POS_KEY: []}
                
    def _get_seqs_and_targets(self, sequence):
        seqs, targets = [], []
        if len(sequence.shape) == 1:
            padding = np.zeros((self.seq_len))
            sequence = np.concatenate((padding, sequence), axis=0)
        else:
            padding = np.zeros((self.seq_len, sequence.shape[1]))
            sequence = np.concatenate((padding, sequence), axis=0)
        # sequence = np.concatenate((padding, sequence), axis=1)
        for i in range(sequence.shape[0] - self.seq_len):
            seqs.append(sequence[i:(i + self.seq_len)])
            if self.train_type == 'next_step':
                targets.append(sequence[i + self.seq_len])
            elif self.train_type == 'full_sequence': 
                targets.append(sequence[(i + 1):(i + self.seq_len + 1)])
        return seqs, targets

    def __len__(self):
        """
        The length of the dataset.
        :return: the number of sequences in the dataset
        """
        return len(self.sequences['pitch_numbers'])

    def __getitem__(self, index):
        """
        A sequence and its target.
        :param index: the index of the sequence and target to fetch
        :return: the sequence and target at the specified index
        """
        seqs = {k: torch.LongTensor(seqs[index]) for k, seqs in self.sequences.items()}
        seqs[const.CHORD_KEY] = seqs[const.CHORD_KEY].float()
        targets = {k: torch.LongTensor(np.array(targs[index])) for k, targs in self.targets.items()}
        targets[const.CHORD_KEY] = targets[const.CHORD_KEY].float()
        return (seqs, targets)
