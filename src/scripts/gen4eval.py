import argparse
import glob
import os
import os.path as op
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-pn', '--pitch_run_name', type=str, default="ICCC_Bebop-Feb07",
                    help="select which pitch run to use")
parser.add_argument('-dn', '--dur_run_name', type=str, default="ICCC_Bebop-Feb07",
                    help="select which dur run to use")
parser.add_argument('-nr', '--num_repeats', type=int, default=1,
                    help="if you want to generate more than 1 run through of a song")
parser.add_argument('-sl', '--seed_length', type=int, default=10,
                    help="number of measures to use as seeds to the network")
args = parser.parse_args()

root_dir = str(Path(op.abspath(__file__)).parents[2])
model_dir = op.join(root_dir, "src", "models")
data_song_dir = op.join(root_dir, "data", "processed", "songs")

models = [("nc", "no_cond"), 
          ("ic", "inter_cond"), 
          ("bc", "barpos_cond"), 
          ("cc", "chord_cond"),
          ("nxc", "nxt_chord_cond"),
          ("cnc", "chord_nxtchord_cond"),
          ("cic", "chord_inter_cond"), 
          ("cnic", "chord_nxtchord_inter_cond"), 
          ("cbc", "chord_barpos_cond"),
          ("cnbc", "chord_nxtchord_barpos_cond"),
          ("ibc", "inter_barpos_cond"),
          ("cibc", "chord_inter_barpos_cond"),
          ("cnibc", "chord_nxtchord_inter_barpos_cond")]

songs = [op.basename(s) for s in glob.glob(op.join(data_song_dir, '*_0.pkl'))]
# songs = [op.basename(s) for s in os.listdir(data_song_dir)]
for abrv, name in models:
    print("#"*30 + "\n{}\n".format(name) + "#"*30)
    for i, song in enumerate(songs):
        outname = "_".join(["4eval", 'Bebop', song.split('.')[0]])
        try:
            subprocess.call(['python', 'make_melody.py', '-m', abrv,  
                             '-pn', args.pitch_run_name, '-dn', args.dur_run_name,  
                             '-ss', song, '-sl', str(args.seed_length), '-t', outname])
        except RuntimeError:
            continue
        if i > 0:
            break
