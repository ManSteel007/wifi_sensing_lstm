from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers import BatchNormalization
from keras.utils import to_categorical
import numpy as np 
import tensorflow as tf
import glob
import os
import csv
from keras.optimizers import Adam, Nadam, SGD
from keras import regularizers
from keras.layers import LeakyReLU
from tensorflow_addons.optimizers import CyclicalLearningRate

class CSIModelConfig:

    def __init__(self, win_len=300, step=50, thrshd=0.6):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("lie down", "fall", "bend", "run", "sitdown", "standup", "walk")
        # self._downsample = downsample

    def preprocessing(self, raw_folder, save=False):

        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        return numpy_tuple
    
    def load_csi_data_from_files(self, np_files):

        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for lie down, fall, bend, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        # if self._downsample > 1:
        #     x = [arr[:,::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:,i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)


def extract_csi(raw_folder, labels, save=False, win_len=300, thrshd=0.6, step=50):

    ans = []
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    return tuple(ans)



def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=300, thrshd=0.6, step=50):

    print('Starting Extract CSI for Label {}'.format(label))
    label = label.lower()
    if not label in labels:
        raise ValueError("The label {} should be among 'lie down','fall','bend','run','sitdown','standup','walk'".format(labels))
    
    data_path_pattern = os.path.join(raw_folder,label, 'user_*' + label + '*.csv')
    input_csv_files = sorted(glob.glob(data_path_pattern))
    # annot_csv_files = [os.path.basename(fname).replace('user_', 'annotation_user') for fname in input_csv_files]
    # annot_csv_files = [os.path.join(raw_folder, label, fname) for fname in annot_csv_files]
    annot_csv_files = os.path.join(raw_folder,label, 'Annotation_user_*' + label + '*.csv')
    annot_csv_files = sorted(glob.glob(annot_csv_files))
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100,label))
    
    feat_arr = np.concatenate(feature, axis=0)
    if save:
        np.savez_compressed("X_{}.npz".format(label), feat_arr)
    # one hot
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.8, seed=200):

    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len,7))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:],...])
        tmpy = np.zeros((x_arr.shape[0]-split_len,7))
        tmpy[:, i] = 1
        y_valid.append(tmpy)
    
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid

def merge_csi_label(csifile, labelfile, win_len=300, thrshd=0.6, step=50):

    activity = []
    with open(labelfile, 'r') as labelf:
        reader = csv.reader(labelf)
        for line in reader:
            label  = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        reader = csv.reader(csif)
        for line in reader:
            line_array = np.array([float(v) for v in line])
            # extract the amplitude only
            line_array = line_array[0:52]
            csi.append(line_array[np.newaxis,...])
    csi = np.concatenate(csi, axis=0)
    assert(csi.shape[0] == activity.shape[0])
    # screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        cur_activity = activity[index:index+win_len]
        if np.sum(cur_activity)  <  thrshd * win_len:
            index += step
            continue
        cur_feature = np.zeros((1, win_len, 52))
        cur_feature[0] = csi[index:index+win_len, :]
        feature.append(cur_feature)
        index += step
    return np.concatenate(feature, axis=0)

cfg = CSIModelConfig(win_len=300, step=50, thrshd=0.6)
numpy_tuple = cfg.preprocessing('Dataset', save=True)

x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
x_train, y_train, x_valid, y_valid = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk),train_portion=0.8, seed=200)


# fit and evaluate a model

verbose, epochs, batch_size = 0, 300, 64
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
model = Sequential()
model.add(LSTM(128, input_shape=(n_timesteps,n_features)))
# model.add(LSTM(150, input_shape=(n_timesteps,n_features),return_sequences=True)) use reurn sequence if u add more lstm
# model.add(LSTM(64))
# model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(n_outputs, activation='softmax'))
opt= Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history=model.fit(
        x_train,
        y_train,verbose=1,
        batch_size=64, epochs=300,
        validation_split=0.2,
        steps_per_epoch=60,
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_lstm.hdf5',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                save_weights_only=False)
            ])
# evaluate model
_, accuracy = model.evaluate(x_valid, y_valid, batch_size=batch_size, verbose=0)


# summarize scores
model.summary()

    # load the best model
# model = cfg.load_model('best_conv.hdf5')
y_pred = model.predict(x_valid)

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
cm=confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1), normalize='true')
print(cm)

#plot curves
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
