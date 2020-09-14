import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, LSTM, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pre_define import *

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #(or "1" or "2")





# Predict
def pred(x, y, model_path):
    # Evaluate
    model = load_model(model_path)
    # model.evaluate(x_test, y_test, verbose=1)
    # model.summary()

    print('model_path', model_path, '\n')

    preds = model.predict(x)
    y_preds = np.array([1 if pred[0] > 0.5 else 0 for pred in preds])
    # print('y', y)
    # print('preds', preds)
    # print('y_preds', y_preds)
    acc = np.sum(y == y_preds)
    total = len(y)
    print('acc', acc, 'total', total, acc/total)


    print('Confusion Matrix')
    C = confusion_matrix(y, y_preds)
    Cm = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    print(C)
    print('C.astype(np.float).sum(axis=1)', C.astype(np.float).sum(axis=1))
    print(Cm)
    # print('Classification Report')
    # print(classification_report(y, y_preds))
    print('\n')



x = np.concatenate((x_train, x_val))
y = np.concatenate((y_train, y_val))

model_paths = ['output/cnn_best__9635_1778.h5', 'output/lstm_best__9427_1926.h5']
for model_path in model_paths:
    print('* All result:')
    pred(x, y, model_path)
    print('* Test result:')
    pred(x_test, y_test, model_path)
    print('--------------------------\n')