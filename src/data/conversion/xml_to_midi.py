import os
import os.path as op
import subprocess
from pathlib import Path

def main():
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    xml_dir = op.join(root_dir, 'data', 'raw', 'xml')

    fnames = os.listdir(xml_dir)
    fdicts = [{'basename': fname.split('.')[0], 
               'path': op.join(xml_dir, fname)} for fname in fnames]

    for fdict in fdicts:
        outpath = op.join(root_dir, 'data', 'raw', 'midi', fdict['basename'] + '.mid')
        mscore_path = '/Applications/MuseScore 2.app/Contents/MacOS/mscore'
        subprocess.call([mscore_path, fdict['path'], '-o', outpath])
    return

if __name__ == '__main__':
    main()
