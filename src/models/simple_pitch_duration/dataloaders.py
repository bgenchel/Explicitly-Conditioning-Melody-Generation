import numpy as np
import pdb
from torch.utils.data import DataLoader


class LeadSheetDataLoader(DataLoader):
    """
    DataLoader class for lead sheet data parsed from music xml.
    """
    def __init__(self, dataset, num_songs=None, **kwargs):
        # super(LeadSheetDataLoader, self).__init__(**kwargs)
        """
        Initializes the class, defines the number of batches and other parameters
        Args:
            dataset:    dict carrying both train and validation sets
            num_songs:	int, number of songs for the training data
        """
        train, valid = dataset['train'], dataset['valid']
        if num_songs is None:
            num_songs = len(train)
        # check if input parameters are accurate
        assert num_songs <= len(train)
        self.train = train[:num_songs]
        self.valid = valid
        self.num_songs = num_songs

    def get_pitch_seqs(self, seq_len=2, target_as_vector=False):
        train_pitch_seqs = []
        train_next_pitches = []
        valid_pitch_seqs = []
        valid_next_pitches = []
        for song in self.train:
            song_pitches = []
            for measure in song['measures']:
                for i, pitch in enumerate(measure['pitch_numbers']):
                    song_pitches.append(pitch)
            for i in range(0, len(song_pitches) - seq_len):
                train_pitch_seqs.append(np.array(song_pitches[i:i+seq_len]))
                if target_as_vector:
                    next_pitch = np.zeros(128)
                    next_pitch[song_pitches[i+seq_len]] = 1
                    train_next_pitches.append(next_pitch)
                else:
                    train_next_pitches.append(song_pitches[i+seq_len])

        for song in self.valid:
            song_pitches = []
            for measure in song['measures']:
                for i, pitch in enumerate(measure['pitch_numbers']):
                    song_pitches.append(pitch)
            for i in range(0, len(song_pitches) - seq_len):
                valid_pitch_seqs.append(np.array(song_pitches[i:i+seq_len]))
                if target_as_vector:
                    next_pitch = np.zeros(128)
                    next_pitch[song_pitches[i+seq_len]] = 1
                    valid_next_pitches.append(next_pitch)
                else:
                    valid_next_pitches.append(song_pitches[i+seq_len])
                    
        return (np.array(train_pitch_seqs), np.array(train_next_pitches), 
                np.array(valid_pitch_seqs), np.array(valid_next_pitches))

    def get_dur_seqs(self, seq_len=2, target_as_vector=False):
        train_dur_seqs = []
        train_next_durs = []
        valid_dur_seqs = []
        valid_next_durs = []
        for song in self.train:
            song_durs = []
            for measure in song['measures']:
                for i, dur_tag in enumerate(measure['duration_tags']):
                    song_durs.append(dur_tag)
            for i in range(0, len(song_durs) - seq_len):
                train_dur_seqs.append(np.array(song_durs[i:i+seq_len]))
                if target_as_vector:
                    next_dur = np.zeros(18)
                    next_dur[song_durs[i+seq_len]] = 1
                    train_next_durs.append(next_dur)
                else:
                    train_next_durs.append(song_durs[i+seq_len])

        for song in self.valid:
            song_durs = []
            for measure in song['measures']:
                for i, dur in enumerate(measure['duration_tags']):
                    song_durs.append(dur)
            for i in range(0, len(song_durs) - seq_len):
                valid_dur_seqs.append(np.array(song_durs[i:i+seq_len]))
                if target_as_vector:
                    next_dur = np.zeros(128)
                    next_dur[song_durs[i+seq_len]] = 1
                    valid_next_durs.append(next_dur)
                else:
                    valid_next_durs.append(song_durs[i+seq_len])

        return (np.array(train_dur_seqs), np.array(train_next_durs),
                np.array(valid_dur_seqs), np.array(valid_next_durs))

    def _get_batched_seqs(self, seqs, targets, batch_size=1):
        assert batch_size <= len(seqs)
        num_batches = int(np.floor(len(seqs) / batch_size))
        batched_seqs = np.split(seqs[:num_batches*batch_size], num_batches, axis=0)
        batched_targets = np.split(targets[:num_batches*batch_size], num_batches, axis=0)
        return np.array(batched_seqs), np.array(batched_targets)

    def get_batched_pitch_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        pitch_seqs = self.get_pitch_seqs(seq_len, target_as_vector)
        train_seqs, train_targets, valid_seqs, valid_targets = pitch_seqs
        batched_train_seqs, batched_train_targets = self._get_batched_seqs(train_seqs,
                                                                           train_targets,
                                                                           batch_size)
        batched_valid_seqs, batched_valid_targets = self._get_batched_seqs(valid_seqs,
                                                                           valid_targets,
                                                                           batch_size)
        return (batched_train_seqs, batched_train_targets, 
                batched_valid_seqs, batched_valid_targets)

    def get_batched_dur_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        dur_seqs = self.get_dur_seqs(seq_len, target_as_vector)
        train_seqs, train_targets, valid_seqs, valid_targets = dur_seqs
        batched_train_seqs, batched_train_targets = self._get_batched_seqs(train_seqs,
                                                                           train_targets,
                                                                           batch_size)
        batched_valid_seqs, batched_valid_targets = self._get_batched_seqs(valid_seqs,
                                                                           valid_targets,
                                                                           batch_size)
        return (batched_train_seqs, batched_train_targets, 
                batched_valid_seqs, batched_valid_targets)
