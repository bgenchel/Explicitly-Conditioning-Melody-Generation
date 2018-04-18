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
                dataset:  	object of the RawAudioDataset class, should be properly initialized
                num_data_pts:	int, number of data points to be considered while loading the data
        """
        if num_songs is None:
            num_songs = len(dataset)

        # check if input parameters are accurate
        assert num_songs <= len(dataset)
        self.dataset = dataset[:num_songs]
        self.num_songs = num_songs

    def get_pitch_seqs(self, seq_len=2, target_as_vector=False):
        pitch_seqs = []
        next_pitches = []
        for song in self.dataset:
            song_pitches = []
            for measure in song['measures']:
                for i, pitch in enumerate(measure['pitch_numbers']):
                    song_pitches.append(pitch)
            for i in range(0, len(song_pitches) - seq_len):
                pitch_seqs.append(np.array(song_pitches[i:i+seq_len]))
                if target_as_vector:
                    next_pitch = np.zeros(128)
                    next_pitch[song_pitches[i+seq_len]] = 1
                    next_pitches.append(next_pitch)
                else:
                    next_pitches.append(song_pitches[i+seq_len])

        return np.array(pitch_seqs), np.array(next_pitches)

    def get_dur_seqs(self, seq_len=2, target_as_vector=False):
        dur_seqs = []
        next_durs = []
        for song in self.dataset:
            song_durs = []
            for measure in song['measures']:
                for i, dur_tag in enumerate(measure['duration_tags']):
                    song_durs.append(dur_tag)
            for i in range(0, len(song_durs) - seq_len):
                dur_seqs.append(np.array(song_durs[i:i+seq_len]))
                if target_as_vector:
                    next_dur = np.zeros(18)
                    next_dur[song_durs[i+seq_len]] = 1
                    next_durs.append(next_durs)
                else:
                    next_durs.append(song_durs[i+seq_len])

        return np.array(dur_seqs), np.array(next_durs)

    def get_batched_pitch_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        pitch_seqs, next_pitches = self.get_pitch_seqs(seq_len, target_as_vector)
        assert batch_size <= len(pitch_seqs)
        num_batches = int(np.floor(len(pitch_seqs) / batch_size))
        batched_pitch_seqs = np.split(pitch_seqs[:num_batches*batch_size], num_batches, axis=0)
        batched_next_pitches = np.split(next_pitches[:num_batches*batch_size], num_batches, axis=0)
        return np.array(batched_pitch_seqs), np.array(batched_next_pitches)

    def get_batched_dur_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        dur_seqs, next_durs = self.get_dur_seqs(seq_len, target_as_vector)
        assert batch_size <= len(dur_seqs)
        num_batches = int(np.floor(len(dur_seqs) / batch_size))
        batched_dur_seqs = np.split(dur_seqs[:num_batches*batch_size], num_batches, axis=0)
        batched_next_durs = np.split(next_durs[:num_batches*batch_size], num_batches, axis=0)
        return np.array(batched_dur_seqs), np.array(batched_next_durs)
