import pickle
from configparser import ConfigParser
from pathlib import Path

import innvestigate
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

config = ConfigParser()
config.read('config.INI')

pickle_path = Path('./experiments/pickle')
model_path = Path('./experiments/model')

input_shape = pickle.load(open(pickle_path / 'input_shape.p', 'rb'))
filters = config.getint('NETWORK', 'filters')
n_grams = config.getint('NETWORK', 'n_gram')
vector_dim = input_shape[1]
dropout_1 = config.getfloat('NETWORK', 'dropout_1')
dense_neurons = config.getint('NETWORK', 'dense_neurons')
dropout_2 = config.getfloat('NETWORK', 'dropout_2')

model = Sequential()
model.add(Conv2D(filters, kernel_size=(n_grams, vector_dim),
                 activation='relu',
                 input_shape=input_shape,
                 name='conv2d'))
model.add(MaxPooling2D(pool_size=(model.get_layer('conv2d').output_shape[1], 1)))
model.add(Dropout(dropout_1))
model.add(Flatten())
model.add(Dense(dense_neurons, activation='relu'))
model.add(Dropout(dropout_2))
model.add(Dense(1))  # on purpose no activation function

model.load_weights(model_path / 'trained_model.hdf5')

analyser_name = config.get('ANALYSER', 'analyser_name')
analyser = innvestigate.create_analyzer(analyser_name, model)

x_train = pickle.load(open(pickle_path / 'x_train.p', 'rb'))
x_test = pickle.load(open(pickle_path / 'x_test.p', 'rb'))

test_pred = model.predict(x_test)

analyser.fit(x_train)
analysis = analyser.analyze(x_test)

pickle.dump(test_pred, open(pickle_path / 'test_pred.p', 'wb'))
pickle.dump(analysis, open(pickle_path / 'analysis.p', 'wb'))
