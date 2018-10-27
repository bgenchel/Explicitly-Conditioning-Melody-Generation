import argparse
import json
import os
import os.path as op
import torch

def get_args(default_title=""):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--title', default=default_title, type=str,
                        help="custom title for run data directory")
    parser.add_argument('-cp', '--charlie_parker', action="store_true",
                        help="use the charlie parker dataset.")
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help="number of training epochs")
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        help="number of training epochs")
    parser.add_argument('-sl', '--seq_len', default=1, type=int,
                        help="number of previous steps to consider in prediction.")
    parser.add_argument('-hd', '--hidden_dim', default=128, type=int,
                        help="size of hidden state.")
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help="learning rate for sgd")
    parser.add_argument('-do', '--dropout', default=0.0, type=float,
                        help="drop out rate for LSTM")
    parser.add_argument('-bn', '--batch_norm', action="store_true",
                        help="use batch normalization.")
    parser.add_argument('-m', '--model', required=True, choices=("pitch", "duration"), 
                        type=str, help="which model to train.")
    parser.add_argument('-nc', '--no_cuda', action="store_true",
                        help="don't allow the use of CUDA, even if it's available.")
    parser.add_argument('-k', '--keep', action='store_true',
                        help="save model files and other about this run")
    return parser.parse_args()

def save_run(dirpath, info_dict, train_losses, valid_losses, model_inputs, model, keep=False):
    if not op.exists(dirpath):
        os.makedirs(dirpath)

    print('Writing run info file ...')
    with open(op.join(dirpath, 'info.txt'), 'w') as fp:
        max_kw_len = max([len(key) for key in info_dict.keys()])
        for k, v in info_dict.items():
            space_buffer = ' '*(max_kw_len - len(k))
            fp.write('%s:%s\t %s\n'%(str(k), space_buffer, str(v)))
        fp.close()

    if keep:
        print('Writing training losses ...') 
        json.dump(train_losses, open(op.join(dirpath, 'train_losses.json'), 'w'), indent=4)

        print('Writing validation losses ...') 
        json.dump(valid_losses, open(op.join(dirpath, 'valid_losses.json'), 'w'), indent=4)

        print('Saving model ...')
        json.dump(model_inputs, open(op.join(dirpath, 'model_inputs.json'), 'w'), indent=4)
        torch.save(model.state_dict(), op.join(dirpath, 'model_state.pt'))
    return
