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
    parser.add_argument('-sl', '--seq_len', default=32, type=int,
                        help="number of previous steps to consider in prediction.")
    parser.add_argument('-hd', '--hidden_dim', default=256, type=int,
                        help="size of hidden state.")
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help="learning rate for sgd")
    parser.add_argument('-do', '--dropout', default=0.0, type=float,
                        help="drop out rate for LSTM")
    parser.add_argument('-bn', '--batch_norm', action="store_true",
                        help="use batch normalization.")
    parser.add_argument('-m', '--model', required=True, choices=("pitch", "duration"), 
                        type=str, help="which model to train.")
    parser.add_argument('-tt', '--train_type', default="full_sequence", choices=("next_step", "full_sequence"), 
                        type=str, help="How to train the model / calculate loss.")
    parser.add_argument('-nc', '--no_cuda', action="store_true",
                        help="don't allow the use of CUDA, even if it's available.")
    parser.add_argument('-cd', '--cuda_device', type=int, default=0, choices=(0, 1),
                        help="Which GPU to use.")
    parser.add_argument('-k', '--keep', action='store_true',
                        help="save model files and other about this run")
    return parser.parse_args()
