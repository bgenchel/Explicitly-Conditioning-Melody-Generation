import numpy as np
from torch.utils.data import DataLoader
from .constants import PITCH_DIM, DUR_DIM, NOTES_MAP


class LeadSheetDataLoader(DataLoader):
    """
    DataLoader class for lead sheet data parsed from music xml.
    """

    def __init__(self, dataset, num_songs=None, **kwargs):
        """
        Initializes the class, defines the number of batches and other parameters
        Args:
                dataset:  	object of the RawAudioDataset class, should be properly initialized
                num_data_pts:	int, number of data points to be considered while loading the data
        """
        super(LeadSheetDataLoader, self).__init__(self, **kwargs)
        train, valid = dataset['train'], dataset['valid']
        if num_songs is None:
            num_songs = len(train)
        # check if input parameters are accurate
        assert num_songs <= len(train)
        self.train = train[:num_songs]
        self.valid = valid
        self.num_songs = num_songs

    def _get_seqs(self, seq_key="", seq_len=2, target_as_vector=False, target_vector_size=-1):
        def extract_sequences(dataset):
            seqs = []
            targets = []
            for song in dataset:
                full_song_seq = []
                for measure in song['measures']:
                    for i, thing in enumerate(measure[seq_key]):
                        full_song_seq.append(thing)

                full_song_seq = [NOTES_MAP['rest']]*seq_len + full_song_seq
                for i in range(0, len(full_song_seq) - seq_len):
                    seqs.append(np.array(full_song_seq[i:i+seq_len]))

                    if target_as_vector:
                        target = np.zeros(target_vector_size)
                        target[full_song_seq[i + seq_len]] = 1
                        targets.append(target)
                    else:
                        targets.append(full_song_seq[i+seq_len])
            return seqs, targets

        train_seqs, train_targets = extract_sequences(self.train)
        valid_seqs, valid_targets = extract_sequences(self.valid)
        return (train_seqs, train_targets, valid_seqs, valid_targets)

    def get_harmony(self, seq_len=2):
        def extract_note_chord_seqs(dataset):
            seqs = []
            targets = []
            for song in dataset:
                note_chords = [] # the chord playing at each note
                for measure in song['measures']:
                    chord_index = 0
                    for i in range(len(measure['pitch_numbers'])):
                        if (i == measure['half_index']) and (len(measure['harmonies']) > 1):
                            chord_index += 1
                        note_chords.append(measure['harmonies'][chord_index])

                blank_chord = lambda: [0]*12
                note_chords = [blank_chord() for _ in range(seq_len)] + note_chords
                for i in range(0, len(note_chords) - seq_len):
                    seqs.append(np.array(note_chords[i:i+seq_len]))
                    targets.append(np.array(note_chords[i+seq_len]))
            return seqs, targets

        train_seqs, train_targets = extract_note_chord_seqs(self.train)
        valid_seqs, valid_targets = extract_note_chord_seqs(self.valid)
        return (train_seqs, train_targets, valid_seqs, valid_targets)

    def _get_batched(self, seqs_getter, seq_len=2, batch_size=1, target_as_vector=False):
        (train_seqs, train_targets, valid_seqs, valid_targets) = seqs_getter(
            seq_len=seq_len, target_as_vector=target_as_vector)

        def batch(seqs, targets):
            assert len(seqs) == len(targets)
            assert batch_size <= len(seqs) # can we even batch?
            num_batches = int(np.floor(len(seqs) / batch_size))
            batched_seqs = np.split(np.array(seqs[:num_batches*batch_size]),
                num_batches, axis=0)
            batched_targets = np.split(np.array(targets[:num_batches*batch_size]),
                num_batches, axis=0)
            return batched_seqs, batched_targets

        batched_train_seqs, batched_train_targets = batch(train_seqs, train_targets)
        batched_valid_seqs, batched_valid_targets = batch(valid_seqs, valid_targets)
        return {'batched_train_seqs': batched_train_seqs,
                'batched_train_targets': batched_train_targets,
                'batched_valid_seqs': batched_valid_seqs,
                'batched_valid_targets': batched_valid_targets}

    def get_batched_harmony(self, seq_len=2, batch_size=1):
        return self._get_batched(self.get_harmony, seq_len, batch_size)

    def get_pitch_seqs(self, seq_len=2, target_as_vector=False):
        train_seqs, train_targets, valid_seqs, valid_targets = self._get_seqs(
            "pitch_numbers", seq_len, target_as_vector, PITCH_DIM)
        return (train_seqs, train_targets, valid_seqs, valid_targets)

    def get_batched_pitch_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        return self._get_batched(self.get_pitch_seqs, seq_len, batch_size, target_as_vector)

    def get_dur_seqs(self, seq_len=2, target_as_vector=False):
        train_seqs, train_targets, valid_seqs, valid_targets = self._get_seqs(
            "duration_tags", seq_len, target_as_vector, DUR_DIM)
        return (train_seqs, train_targets, valid_seqs, valid_targets)

    def get_batched_dur_seqs(self, seq_len=2, batch_size=1, target_as_vector=False):
        return self._get_batched(self.get_dur_seqs, seq_len, batch_size, target_as_vector)
