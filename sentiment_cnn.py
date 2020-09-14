"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

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
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix #, plot_confusion_matrix
import matplotlib.pyplot as plt
from pre_define import *


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #(or "1" or "2")

# Fixed error Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)



# Training parameters
batch_size = 64
num_epochs = 30



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

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])


model.summary()


# Train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# mc = ModelCheckpoint('output/cnn_best.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
mc = ModelCheckpoint('output/cnn_best.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val), callbacks=[es, mc], verbose=2)
model.save('output/model_cnn.h5')


# Evaluate
model = load_model('output/cnn_best.h5')
# model.evaluate(x_test, y_test, verbose=1)



# Predict
def pred(x, y):
    preds = model.predict(x)
    y_preds = np.array([1 if pred[0] > 0.5 else 0 for pred in preds])
    # print('preds', preds)
    # print('y', y)
    # print('y_preds', y_preds)

    acc = np.sum(y == y_preds)
    total = len(y)
    print('acc', acc, 'total', total, acc/total)

    tot_bgn = np.sum(y == 0)
    tot_mal = np.sum(y == 1)

    # # far
    # false_bgn = np.sum([(y != y_preds) and (y == 1) and (y_preds == 0)])
    # print('false_bgn', false_bgn, 'tot_bgn', tot_bgn, false_bgn/tot_bgn)
    # # tpr
    # correct_mal = np.sum([(y and 1) and (y_preds and y)] == 1)
    # print('correct_mal', correct_mal, 'tot_mal', tot_mal, correct_mal/tot_mal)

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

    # titles_options = [("Confusion matrix, without normalization", None),
    #                     ("Normalized confusion matrix", 'true')]
    # for title, normalize in titles_options:
    #     disp = plot_confusion_matrix(model, x, y,
    #                                     display_labels=[0,1],
    #                                     # cmap=plt.cm.Blues,
    #                                     normalize=normalize)
    #     disp.ax_.set_title(title)

    #     print(title)
    #     print(disp.confusion_matrix)

    # plt.show()


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