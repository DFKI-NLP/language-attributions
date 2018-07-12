import datetime
import json
import pickle
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np

pickle_path = Path('./experiments/pickle')
summary_path = Path('./experiments/summary')


def summary(pos_k_idxs, neg_k_idxs, attributions, sentiment, indices, prediction, vocab):
    tokens = []
    for idx in indices:
        tokens.append(vocab[idx])
    assert (len(tokens) == len(attributions))
    top_k_tokens = []
    top_k_idxs = pos_k_idxs if prediction >= 0 else neg_k_idxs
    for k_idx in top_k_idxs:
        top_k_tokens.append(tokens[k_idx])
    attributions_rounded = [round(att, ndigits=3) for att in attributions]
    token_attributions = [str(item) for item in zip(tokens, attributions_rounded)]
    entry = {'sentiment': str(sentiment), 'prediction': str(prediction), 'top_k_tokens': top_k_tokens,
             'attributions': token_attributions}
    return entry


def write_long_json_summary(no, top_k):
    input_shape = pickle.load(open(pickle_path / 'input_shape.p', 'rb'))

    analysis = pickle.load(open(pickle_path / 'analysis.p', 'rb'))
    analysis_sum = [np.sum(ana.reshape(input_shape[0], input_shape[1]).transpose(), axis=0) for ana in analysis]
    analysis_pos_top_k_idxs = [np.argsort(-ana_sum)[:top_k] for ana_sum in analysis_sum]
    analysis_neg_top_k_idxs = [np.argsort(ana_sum)[:top_k] for ana_sum in analysis_sum]

    encoder_vocab = pickle.load(open(pickle_path / 'encoder_vocab.p', 'rb'))

    test_indices_and_sentiments = pickle.load(open(pickle_path / 'test_indices_and_sentiment.p', 'rb'))

    pred_test = pickle.load(open(pickle_path / 'test_pred.p', 'rb'))

    h = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"]).strip()
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    long_summary = {'git_hash': str(h), 'time_stamp': str(st)}

    for idx in range(min(len(analysis_sum[0]), no)):
        json_summary = summary(analysis_pos_top_k_idxs[idx],
                               analysis_neg_top_k_idxs[idx],
                               analysis_sum[idx],
                               test_indices_and_sentiments[idx][1],
                               test_indices_and_sentiments[idx][0],
                               pred_test[idx][0],
                               encoder_vocab)
        long_summary['input_{}'.format(idx)] = json_summary
    
    summary_path.mkdir(exist_ok=True)
    with open(summary_path / 'summary.json', 'w') as outfile:
        json.dump(long_summary, outfile, indent=4, separators=(',', ': '))


config = ConfigParser()
config.read('config.INI')
k = config.getint('ANALYSER', 'top_k')
summary_length = config.getint('ANALYSER', 'summary_length')
write_long_json_summary(summary_length, k)
