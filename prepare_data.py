import pickle
from configparser import ConfigParser
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Embedding
from torchnlp.word_to_vector import GloVe

from spacy_encoder import SpacyEncoder


def get_data(data_frame):
    data = []
    for idx, row in data_frame.iterrows():
        if idx == 0:
            continue
        sentiment = (int(row['sentiment']) - 1)
        title = row['title']
        review = row['review']
        if title != title:
            title = ""
        if review != review:
            review = ""
        data.append((sentiment, title + " : " + review))
    return data


def retrieve_texts(data):
    return [tup[1] for tup in data]


def pad(max_len, idxs):
    if len(idxs) > max_len:
        return idxs[:max_len]
    padding = max_len - len(idxs)
    zeros = list(repeat(0, padding))
    idxs = idxs + zeros
    return idxs


def docs2idxs(corpus, max_len=-1, encoder=None):
    if encoder is None:
        encoder = SpacyEncoder(corpus)
    indices = [encoder.encode(doc).numpy().tolist() for doc in corpus]
    if max_len <= 0:
        max_len = max([len(lst) for lst in indices])
    indices = [pad(max_len, idxs) for idxs in indices]
    return encoder, indices, max_len


def weights(encoder, vectors):
    ws = np.zeros((encoder.vocab_size, vectors.dim))
    for index, word in enumerate(encoder.vocab):
        ws[index] = vectors[word]
    return ws


config = ConfigParser()
config.read('config.INI')

train_csv = config.get('TRAINING', 'train_csv')
test_csv = config.get('TRAINING', 'test_csv')

amazon_training_csv = pd.read_csv(train_csv, header=None,
                                  names=['sentiment', 'title', 'review'])
training_data = get_data(amazon_training_csv)
training_texts = retrieve_texts(training_data)
enc, training_indices, training_seq_len = docs2idxs(training_texts)
training_indices_and_sentiment = [(idxs, d[0]) for (idxs, d) in zip(training_indices, training_data)]

amazon_test_csv = pd.read_csv(test_csv, header=None,
                              names=['sentiment', 'title', 'review'])
test_data = get_data(amazon_test_csv)
test_texts = retrieve_texts(test_data)
_, test_indices, _ = docs2idxs(test_texts, max_len=training_seq_len, encoder=enc)
test_indices_and_sentiment = [(idxs, d[0]) for (idxs, d) in zip(test_indices, test_data)]

training_titles = set(amazon_training_csv['title'])
training_reviews = set(amazon_training_csv['review'])
for idx, item in amazon_test_csv.iterrows():
    if item['review'] in training_reviews:
        if item['title'] in training_titles:
            raise AssertionError("Row w/ title {} redundant.".format(item['title']))

vecs = GloVe(cache=config.get('PREPARATION', 'word_vector_cache'))
embedding_weights = weights(enc, vecs)

embedding_model = Sequential()
embedding_model.add(Embedding(enc.vocab_size,
                              vecs.dim,
                              weights=[embedding_weights],
                              input_length=training_seq_len,
                              trainable=False))
embedding_model.compile('rmsprop', 'mse')

input_shape = (training_seq_len, vecs.dim, 1)

x_train_unshaped = [embedding_model.predict(np.array(sample[0]).reshape(1, -1)) for sample in
                    training_indices_and_sentiment]  # shape n * (1 * seq_len * vector_dim)
x_test_unshaped = [embedding_model.predict(np.array(sample[0]).reshape(1, -1)) for sample in
                   test_indices_and_sentiment]  # shape n * (1 * seq_len * vector_dim)

x_train = [sample.reshape(input_shape) for sample in x_train_unshaped]
x_train = np.array(x_train)
y_train = [sample[1] for sample in training_indices_and_sentiment]
y_train = np.array(y_train)

x_test = [sample.reshape(input_shape) for sample in x_test_unshaped]
x_test = np.array(x_test)
y_test = [sample[1] for sample in test_indices_and_sentiment]
y_test = np.array(y_test)

pickle_path = Path('./experiments/pickle/')
pickle_path.mkdir(exist_ok=True)

pickle.dump(enc.vocab, open(pickle_path / 'encoder_vocab.p', 'wb'))

pickle.dump(training_indices_and_sentiment, open(pickle_path / 'training_indices_and_sentiment.p', 'wb'))
pickle.dump(test_indices_and_sentiment, open(pickle_path / 'test_indices_and_sentiment.p', 'wb'))

pickle.dump(embedding_weights, open(pickle_path / 'embedding_weights.p', 'wb'))
pickle.dump(input_shape, open(pickle_path / 'input_shape.p', 'wb'))

pickle.dump(x_train, open(pickle_path / 'x_train.p', 'wb'))
pickle.dump(y_train, open(pickle_path / 'y_train.p', 'wb'))

pickle.dump(x_test, open(pickle_path / 'x_test.p', 'wb'))
pickle.dump(y_test, open(pickle_path / 'y_test.p', 'wb'))
