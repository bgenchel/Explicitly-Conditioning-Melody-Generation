import argparse
import glob
import os.path as op
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default="condition_everything", type=str,
                    choices=('condition_everything', 'chord_conditioned_pitch',
                    'simple_pitch_duration'), help="which model to use for generation")
parser.add_argument('-pn', '--pitch_run_name', type=str,
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str,
                    help="select which dur run to use")
parser.add_argument('-sm', '--seed_measures', type=int, default=1,
                    help="number of measures to use as seeds to the network")
args = parser.parse_args()

root_dir = str(Path(op.abspath(__file__)).parents[2])
data_song_dir = op.join(root_dir, "data", "processed", "songs")
songs = [op.basename(s) for s in glob.glob(op.join(data_song_dir, 'charlie_parker*_0.pkl'))]

for song in songs:
    print(song)
    outname = "_".join(["4eval", song.split('.')[0]])
    try:
        subprocess.call(['python', 'make_melody.py', '-m', args.model,  
                         '-pn', args.pitch_run_name, '-dn', args.dur_run_name,  
                         '-ss', song, '-sm', args.seed_measures, '-t', outname])
    except RuntimeError:
        continue
