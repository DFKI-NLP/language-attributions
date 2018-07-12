import pickle
from configparser import ConfigParser
from pathlib import Path

import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import f1_score


def train_dev_split(data, dev=.2):
    assert (0 <= dev <= 1)
    data_size = len(data)
    dev_size = int(data_size * dev)
    train, dev = data[:(data_size - dev_size)], data[(data_size - dev_size):]
    return train, dev


config = ConfigParser()
config.read('config.INI')

input_shape = pickle.load(open('./experiments/pickle/input_shape.p', 'rb'))
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
model.add(Dense(1, activation='sigmoid'))

model_path = Path('./experiments/model')
model_path.mkdir(exist_ok=True)

pickle_path = Path('./experiments/pickle')

callback = keras.callbacks.ModelCheckpoint(str(model_path / 'trained_model.hdf5'), monitor='val_acc', verbose=0,
                                           save_best_only=True, save_weights_only=True, mode='auto', period=1)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.summary()

x_train = pickle.load(open(pickle_path / 'x_train.p', 'rb'))
y_train = pickle.load(open(pickle_path / 'y_train.p', 'rb'))

x_test = pickle.load(open(pickle_path / 'x_test.p', 'rb'))
y_test = pickle.load(open(pickle_path / 'y_test.p', 'rb'))

x_train, x_dev = train_dev_split(x_train, dev=config.getfloat('TRAINING', 'dev_split'))
y_train, y_dev = train_dev_split(y_train, dev=config.getfloat('TRAINING', 'dev_split'))

batch_size = config.getint('TRAINING', 'batch_size')
epochs = config.getint('TRAINING', 'epochs')

model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks=[callback],
          epochs=epochs,
          verbose=1,
          validation_data=(x_dev, y_dev))

pred_test = model.predict(x_test)

pred_test = pred_test >= 0.5

# simple F-score because of neglectable class imbalance
f1 = f1_score(y_test, pred_test)

print("Test F1: {}".format(f1))
