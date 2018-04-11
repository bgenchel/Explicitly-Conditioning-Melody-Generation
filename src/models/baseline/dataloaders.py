import numpy as np
import pdb
from torch.utils.data import DataLoader


class LeadSheetDataLoader(DataLoader):
    """
    DataLoader class for lead sheet data parsed from music xml.
    """

    def __init__(self, dataset, num_songs=None):
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

    def get_harmony(self, seq_len=2):
        harmony_seqs = []
        next_harmonies = []
        for song in self.dataset:
            song_harmonies = [measure['harmonies'][0] for measure in song['measures']]
            for i in range(0, len(song_harmonies) - seq_len):
                harmony_seqs.append(np.ndarray(song_harmonies[i:i + seq_len]))
                next_harmonies.append(np.ndarray(song_harmonies[i + seq_len]))
        return np.ndarray(harmony_seqs), np.ndarray(next_harmonies)

    def get_batched_harmony(self, seq_len=2, batch_size=1):
        harmony_seqs, next_harmonies = self.get_harmony(seq_len)
        assert batch_size <= len(harmony_seqs)
        num_batches = int(np.floor(len(harmony_seqs) / batch_size))
        batched_harmony_seqs = np.split(harmony_seqs, num_batches, axis=0)
        batched_next_harmonies = np.split(next_harmonies, num_batches, axis=0)
        return batched_harmony_seqs, batched_next_harmonies

    def get_pitch_seqs(self, seq_len=2):
        harmony_seqs = []
        pitch_seqs = []
        next_pitches = []
        for song in self.dataset:
            song_pitch_harmonies = []
            song_pitches = []
            for measure in song['measures']:
                harmony_index = 0
                for i, pitch in enumerate(measure['pitch_numbers']):
                    # pdb.set_trace()
                    if (i == measure['half_index']) and (len(measure['harmonies']) > 1):
                        harmony_index += 1
                    song_pitches.append(pitch)
                    song_pitch_harmonies.append(measure['harmonies'][harmony_index])

            for i in range(0, len(song_pitch_harmonies) - seq_len):
                harmony_seqs.append(np.array(song_pitch_harmonies[i:i+seq_len]))
                pitch_seqs.append(np.array(song_pitches[i:i+seq_len]))
                next_pitches.append(np.array(song_pitches[i+seq_len]))

        return np.array(harmony_seqs), np.array(pitch_seqs), np.array(next_pitches)

    def get_batched_pitch_seqs(self, seq_len=2, batch_size=1):
        harmony_seqs, pitch_seqs, next_pitches = self.get_pitch_seqs(seq_len)
        assert batch_size <= len(harmony_seqs)
        num_batches = int(np.floor(len(harmony_seqs) / batch_size))
        batched_harmony_seqs = np.split(harmony_seqs, num_batches, axis=0)
        batched_pitch_seqs = np.split(pitch_seqs, num_batches, axis=0)
        batched_next_pitches = np.split(next_pitches, num_batches, axis=0)
        return np.array(batched_harmony_seqs), np.array(batched_pitch_seqs), np.array(batched_next_pitches)
