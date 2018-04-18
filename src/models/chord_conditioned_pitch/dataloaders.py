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

    # def get_all_seqs(self, seq_len=2, target_as_vector=False): 
    #     pitch_seqs = []
    #     next_pitches = []
    #     dur_seqs = []
    #     next_durs = []
    #     harmony_seqs = []
    #     next_harmonies = []
    #     for song in self.dataset:
    #         song_pitches = []
    #         song_durs = []
    #         song_pitch_harmonies = []
    #         for measure in song['measures']:
    #             harmony_index = 0
    #             for i, pitch in enumerate(measure['pitch_numbers']):
    #                 song_pitches.append(pitch)
    #                 if (i == measure['half_index']) and (len(measure['harmonies']) > 1):
    #                     harmony_index += 1
    #                 song_pitch_harmonies.append(measure['harmonies'][harmony_index])

    #         for i in range(0, len(song_pitches) - seq_len):
    #             pitch_seqs.append(np.array(song_pitches[i:i+seq_len]))

    #             if target_as_vector:
    #                 next_pitch = np.zeros(128)
    #                 next_pitch[song_pitches[i+seq_len]] = 1
    #                 next_pitches.append(next_pitch)
    #             else:
    #                 next_pitches.append(song_pitches[i+seq_len])
                
    #     return np.array(pitch_seqs), np.array(next_pitches)

    def _get_seqs(self, seq_key="", seq_len=2, target_as_vector=False, target_vector_size=-1):
        thing_seqs = []
        next_things = []
        for song in self.dataset:
            song_things = []
            for measure in song['measures']:
                for i, thing in enumerate(measure[seq_key]):
                    song_things.append(thing)

            for i in range(0, len(song_things) - seq_len):
                thing_seqs.append(np.array(song_things[i:i+seq_len]))

                if target_as_vector:
                    next_thing = np.zeros(target_vector_size)
                    next_thing[song_things[i+seq_len]] = 1
                    next_things.append(next_thing)
                else:
                    next_things.append(song_things[i+seq_len])
                
        return np.array(thing_seqs), np.array(next_things)

    def _get_batched(self, seqs_getter, seq_len=2, batch_size=1, target_as_vector=False):
        thing_seqs, next_things = seqs_getter(seq_len, target_as_vector)
        assert batch_size <= len(thing_seqs)
        num_batches = int(np.floor(len(thing_seqs) / batch_size))
        batched_thing_seqs = np.split(thing_seqs[:num_batches*batch_size], num_batches, axis=0)
        batched_next_things = np.split(next_things[:num_batches*batch_size], num_batches, axis=0)
        return batched_thing_seqs, batched_next_things

    def get_harmony(self, seq_len=2, **kwargs):
        harmony_seqs = []
        next_harmonies = []
        for song in self.dataset:
            song_pitch_harmonies = []
            for measure in song['measures']:
                harmony_index = 0
                for i, pitch in enumerate(measure['pitch_numbers']):
                    if (i == measure['half_index']) and (len(measure['harmonies']) > 1):
                        harmony_index += 1
                    song_pitch_harmonies.append(measure['harmonies'][harmony_index])

            for i in range(0, len(song_pitch_harmonies) - seq_len):
                harmony_seqs.append(np.array(song_pitch_harmonies[i:i+seq_len]))
                next_harmonies.append(np.array(song_pitch_harmonies[i+seq_len]))

        return np.array(harmony_seqs), np.array(next_harmonies)

    def get_batched_harmony(self, seq_len=2, batch_size=1):
        batched_harmony_seq, batched_next_harmonie = self._get_batched(self.get_harmony, seq_len, batch_size)
        return batched_harmony_seqs, batched_next_harmonies

    def get_pitch_seqs(self, seq_len=2, target_as_vector=False):
        pitch_seqs, next_pitches = self._get_seqs("pitch_numbers", seq_len, 
                                                  target_as_vector, 128)
        return pitch_seqs, next_pitches

    def get_batched_pitch_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        batched_pitch_seqs, batched_next_pitches = self._get_batched(self.get_pitch_seqs,
                                                                     seq_len, batch_size,
                                                                     target_as_vector)
        return batched_pitch_seqs, batched_next_pitches

    def get_dur_seqs(self, seq_len=2, target_as_vector=False):
        dur_seqs, next_durs = self._get_seqs("dur_tags", seq_len, 
                                             target_as_vector, 18)
        return dur_seqs, next_durs

    def get_batched_dur_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        batched_dur_seqs, batched_next_durs = self._get_batched(self.get_dur_seqs,
                                                                seq_len, batch_size,
                                                                target_as_vector)
        return batched_dur_seqs, batched_next_durs
