import glob
import os
import os.path as op
import pdb

model_dirs = glob.glob(op.join(os.getcwd(), '*_cond'))
for md in model_dirs:
    midi_dir = op.join(md, 'midi')
    for f in os.listdir(midi_dir):
        for ff in os.listdir(op.join(midi_dir, f)):
            os.rename(op.join(midi_dir, f, ff), op.join(midi_dir, f, ff.replace('Folk', 'Bebop')))
        os.rename(op.join(midi_dir, f), op.join(midi_dir, f.replace('Folk', 'Bebop')))

