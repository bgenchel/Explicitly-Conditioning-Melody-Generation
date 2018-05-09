import glob
import os
import os.path as op
import pickle
import subprocess
from pathlib import Path

PITCH_RUN = 'CP_FIRST_GOOD_RUN'
DUR_RUN = 'CP_FIRST_GOOD_RUN'

root_dir = str(Path(op.abspath(__file__)).parents[3])
data_song_dir = op.join(root_dir, "data", "processed", "songs")
songs = [op.basename(s) for s in glob.glob(op.join(data_song_dir, 'charlie_parker*_0.pkl'))]

for song in songs:
    print(song)
    outname = "_".join(["4eval", song.split('.')[0]])
    try:
        subprocess.call(['python', 'make_melody.py', '-pn', PITCH_RUN, '-dn', DUR_RUN, 
                         '-ss', song, '-sm', '1', '-t', outname])
    except RuntimeError:
        continue
