import os
import os.path as op
import subprocess

class CommandNotFound(Exception):
    def __init__(self, message):
        super(CommandNotFound, self).__init__(message)

def main():
    if subprocess.call(["which", "mscore"]) == 0:
        raise CommandNotFound("mscore not found.")

    fnames = os.listdir('mxl')
    fdicts = [{'basename': fname.split('.')[0], 
               'path': op.join(os.getcwd(), 'mxl', fname)} for fname in fnames]

    for fdict in fdicts:
        outpath = op.join(os.getcwd(), 'xml', fdict['basename'] + '.xml')

        mscore_path = '/Applications/MuseScore2.app/Contents/MacOS/mscore'
        subprocess.call([mscore_path, fdict['path'], '-o', outpath])
    return

if __name__ == '__main__':
    main()
