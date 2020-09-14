"""
Train lstmolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""

import numpy as np
np.random.seed(0)

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



# Training parameters
batch_size = 64
num_epochs = 80

dropout_prob = (0.3, 0.4)
hidden_dims = 16


# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = Dropout(dropout_prob[0])(z)

z = LSTM(8, return_sequences=True)(z)
z = LSTM(8)(z)

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)

'''
intermediate_dim = 16
timesteps = 10

model_input = Input(shape=(sequence_length, embedding_dim))

# LSTM encoding
z = LSTM(intermediate_dim, return_sequences=True)(model_input)
z = LSTM(intermediate_dim, return_sequences=False)(z)
z = Dropout(dropout_prob[1])(z)
# z = Dense(intermediate_dim)(z)
'''

model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])


# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])


print("x_train shape:", x_train.shape)



# Train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# mc = ModelCheckpoint('output/lstm_best.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
mc = ModelCheckpoint('output/lstm_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val), callbacks=[es, mc], verbose=2)
# model.save('output/model_lstm.h5')




# Evaluate
model = load_model('output/lstm_best.h5')
# model.evaluate(x_test, y_test, verbose=1)

model.summary()




# Predict
def pred(x, y):
    preds = model.predict(x)
    y_preds = np.array([1 if pred[0] > 0.5 else 0 for pred in preds])
    # print('y', y)
    # print('preds', preds)
    # print('y_preds', y_preds)
    acc = np.sum(y == y_preds)
    total = len(y)
    print('acc', acc, 'total', total, acc/total)


    # # far
    # tot_bgn = np.sum(y == 0)
    # false_bgn = np.sum([(y and 0) and (y_preds or y)] == 0)
    # print('false_bgn', false_bgn, 'tot_bgn', tot_bgn, false_bgn/tot_bgn)
    # # tpr
    # tot_mal = np.sum(y == 1)
    # correct_mal = np.sum([(y and 1) and (y_preds and y)] == 1)
    # print('correct_mal', correct_mal, 'tot_mal', tot_mal, correct_mal/tot_mal)

    # tot_bgn = np.sum(y == 0)
    # tot_mal = np.sum(y == 1)

    # tpr = 0
    # far = 0
    # for k,v in enumerate(y):
    #     # print(k, y[k], v)
    #     if v == 1 and y_preds[k] == 1:
    #         tpr += 1
    #     if v == 0 and y_preds[k] == 1:
    #         far += 1
    # print('tpr', tpr, 'tot_mal', tot_mal, tpr/tot_mal)
    # print('far', far, 'tot_bgn', tot_bgn, far/tot_bgn)

    print('Confusion Matrix')
    C = confusion_matrix(y, y_preds)
    Cm = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    print(C)
    print('C.astype(np.float).sum(axis=1)', C.astype(np.float).sum(axis=1))
    print(Cm)
    # print('Classification Report')
    # print(classification_report(y, y_preds))


    # labels = ['benign', 'malware']
    # cm = confusion_matrix(y, y_preds)
    # print(cm)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()



x = np.concatenate((x_train, x_val))
y = np.concatenate((y_train, y_val))
print('* Train result:')
pred(x, y)
print('* Test result:')
pred(x_test, y_test)
