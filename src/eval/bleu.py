import glob
import itertools
import json
import numpy as np
import os
import os.path as op
import pickle
from collections import OrderedDict
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from pathlib import Path
from tqdm import tqdm
import pdb

TICKS_PER_BEAT = 24
TICKS_PER_MEASURE = 4 * TICKS_PER_BEAT
TICKS_PER_SENTENCE = 8 * TICKS_PER_MEASURE

ABRV_TO_MODEL = OrderedDict({'nc': 'no_cond',
                 'ic': 'inter_cond',
                 'bc': 'barpos_cond', 
                 'cc': 'chord_cond', 
                 'nxc': 'nxt_chord_cond', 
                 'cnc': 'chord_nxtchord_cond', 
                 'cic': 'chord_inter_cond',
                 'cnic': 'chord_nxtchord_inter_cond',
                 'cbc': 'chord_barpos_cond',
                 'cnbc': 'chord_nxtchord_barpos_cond',
                 'ibc': 'inter_barpos_cond',
                 'cibc': 'chord_inter_barpos_cond',
                 'cnibc': 'chord_nxtchord_inter_barpos_cond'})


class BleuScore:

    @classmethod
    def evaluate_bleu_score(cls, predictions, targets, ticks=False, corpus=True):
        """
        Given an array of predicted sequences and ground truths, compute the BLEU score across the sequences.
        :param predictions: an num_sequences x seq_length numpy matrix
        :param targets: an num_sequences x seq_length numpy matrix
        :return: the BLEU score across the corpus of predicted ticks
        """
        if ticks:
            ref_sentences = cls._ticks_to_sentences(targets)
            cand_sentences = cls._ticks_to_sentences(predictions)
        else:
            ref_sentences = [[str(x) for x in seq] for seq in predictions]
            cand_sentences = [[str(x) for x in seq] for seq in targets]

        if corpus: bleu_score = corpus_bleu([[l] for l in ref_sentences], cand_sentences)
        else:
            bleu_score = 0.0
            num_sentences = 0

            for i in tqdm(range(len(ref_sentences))):
                sentence_bleu_score = sentence_bleu(ref_sentences[i], cand_sentences[i])
                print(sentence_bleu_score)
                bleu_score += sentence_bleu_score
                num_sentences += 1

            bleu_score /= num_sentences

        return bleu_score

    @staticmethod
    def _ticks_to_sentences(ticks):
        """
        Given an array of ticks, converts vector values to strings, returning a list of 8 measure "sentence" concatenations.
        :param ticks: an np array of ticks to convert to sentences
        :return: a list of sentences
        """
        sentences = []

        for seq in ticks:
            sentence = []
            for i in range(seq.shape[0]):
                word = ''.join([str(x) for x in seq[i, :]])
                sentence.append(word)
            sentences.append(sentence)

        return sentences


def main():
    print('ENTERED MAIN')
    root_dir = str(Path(op.abspath(__file__)).parents[2])
    model_dir = op.join(root_dir, "src", "models")
    data_song_dir = op.join(root_dir, "data", "processed", "songs")
    
    scores = {}
    songs = [op.basename(s) for s in glob.glob(op.join(data_song_dir, '*_0.pkl'))]
    for abrv, name in ABRV_TO_MODEL.items():
        print(name)
        ref_pns = []
        ref_dts = []
        cand_pns = []
        cand_dts = []
        midi_dir = op.join(model_dir, name, "midi")
        for song in songs:
            gen_ext = "_".join(["4eval", 'Bebop', song.split(".")[0]])
            gen_path = op.join(midi_dir, gen_ext, gen_ext + '_tokens' + ".json")
            if not op.exists(gen_path):
                pdb.set_trace()
                continue

            song_pkl = pickle.load(open(op.join(data_song_dir, song), "rb"))
            song_pns = [list(itertools.chain(*[g['pitch_numbers'] for g in m['groups']])) for m in song_pkl['measures']]
            song_dts = [list(itertools.chain(*[g['duration_tags'] for g in m['groups']])) for m in song_pkl['measures']]

            gen_tokens = json.load(open(gen_path, 'r'))
            gen_pns = gen_tokens['pitch_numbers'] 
            gen_dts = gen_tokens['duration_tags']

            ref_pns.append(list(itertools.chain(*song_pns)))
            ref_dts.append(list(itertools.chain(*song_dts)))
            cand_pns.append(list(itertools.chain(*gen_pns)))
            cand_dts.append(list(itertools.chain(*gen_dts)))

            if len(song_pns) != len(gen_pns):
                print("{}: {} vs {}".format(song, len(song_pns), len(gen_pns)))

        scores[name] = {}
        scores[name]["pitch"] = BleuScore.evaluate_bleu_score(np.array(cand_pns), np.array(ref_pns))
        scores[name]["duration"] = BleuScore.evaluate_bleu_score(np.array(cand_dts), np.array(ref_dts))

    json.dump(scores, open('bleuScores.json', 'w'), indent=4)


if __name__ == '__main__': 
    main()
