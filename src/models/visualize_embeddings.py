import os
import json
import torch
import numpy as np
from collections import namedtuple
from chord_inter_barpos_cond.model_classes import PitchLSTM, DurationLSTM
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import utils.constants as const

cond_type = 'chord_inter_barpos_cond'
model_type = 'duration'
full_path = os.path.join(cond_type, 'runs', model_type, 'ICCC_FinalRun')


with open(full_path + '/model_inputs.json') as f:
    model_input = json.load(f)
model_input = namedtuple('X', model_input.keys())(*model_input.values())

if model_type == 'pitch':
    Model = PitchLSTM
elif model_type == 'duration':
    Model = DurationLSTM

model = Model(
    hidden_dim=model_input.hidden_dim,
    seq_len=model_input.seq_len,
    batch_size=model_input.batch_size,
    dropout=model_input.dropout,
    batch_norm=model_input.batch_norm,
    no_cuda=model_input.no_cuda
)

path_to_state_dict = full_path + '/model_state.pt'
model.load_state_dict(torch.load(path_to_state_dict))
embedding_layer = model.embedding


def get_tsne(embedding_layer):
    w = embedding_layer.weight.data.cpu().numpy()
    tsne = TSNE(n_components=2, verbose=1., perplexity=40, n_iter=300)
    w_embed =tsne.fit_transform(w)
    return w_embed


def get_pca(embedding_layer):
    w = embedding_layer.weight.data.cpu().numpy()
    pca = PCA(n_components=2, whiten=True)
    pca.fit(w)
    pca_embed = pca.transform(w)
    return pca_embed


def plot_scatter_pitch(w_embed):
    num_elements = 89
    num_unique_elements = 13
    label_indices = np.arange(num_elements)
    label_indices[:-1] = label_indices[:-1] % 12
    label_indices[-1] = 12
    cm = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    for idx in range(num_unique_elements):
        x = w_embed[label_indices == idx, 0]
        y = w_embed[label_indices == idx, 1]
        label = const.INVERSE_NOTES_MAP[idx]
        ax.scatter(x, y, c=cm(1.*idx/num_unique_elements), label=label)
    ax.legend()
    plt.show()


def plot_scatter_duration(w_embed):
    num_unique_elements = w_embed.shape[0]
    label_indices = np.arange(num_unique_elements)
    cm = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    for idx in range(num_unique_elements):
        x = w_embed[label_indices == idx, 0]
        y = w_embed[label_indices == idx, 1]
        label = const.REV_DURATIONS_MAP[idx]
        ax.scatter(x, y, c=cm(1. * idx / num_unique_elements), label=label)
    ax.legend()
    plt.show()

plot_scatter_duration(get_tsne(embedding_layer))
plot_scatter_duration(get_pca(embedding_layer))

a = 1

