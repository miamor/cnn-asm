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
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, LSTM, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
np.random.seed(0)

# ---------------------- Parameters section -------------------
#
# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "keras_data_set"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (8,8)
dropout_prob = (0.2, 0.3)
hidden_dims = 8

# Training parameters
batch_size = 32
num_epochs = 30

# Prepossessing parameters
sequence_length = 100
max_words = 1000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------


# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)
        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_val = sequence.pad_sequences(x_val, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"

        return x_train, y_train, x_val, y_val, x_val, y_val, vocabulary_inv
    else:
        x, y, x_test, y_test, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        y = y.argmax(axis=1)
        y_test = y_test.argmax(axis=1)

        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x = x[shuffle_indices]
        y = y[shuffle_indices]
        train_len = int(len(x) * 0.9)
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_val = x[train_len:]
        y_val = y[train_len:]

        return x_train, y_train, x_val, y_val, x_test, y_test, vocabulary_inv


# Data Preparation
print("Load data...")
x_train, y_train, x_val, y_val, x_test, y_test, vocabulary_inv = load_data(data_source)

print('sequence_length', sequence_length)
print('x_val.shape', x_val.shape)
if sequence_length != x_val.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_val.shape[1]

print("x_train shape:", x_train.shape)
print("x_val shape:", x_val.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and lstmert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_val)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_val = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_val])
        print("x_train static shape:", x_train.shape)
        print("x_val static shape:", x_val.shape)

else:
    raise ValueError("Unknown model type")


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
# lstm_blocks = []
# for sz in filter_sizes:
#     # LSTM(intermediate_dim, return_sequences=False)(encode_sentence)
#     lstm = LSTM(sz)(z)
#     lstm_blocks.append(lstm)
# z = Concatenate()(lstm_blocks) if len(lstm_blocks) > 1 else lstm_blocks[0]
z = LSTM(8)(z)

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)

'''
intermediate_dim = 16
timesteps = 10

input = Input(shape=(timesteps, len(vocabulary_inv), embedding_dim,))

# LSTM encoding
lstm = LSTM(intermediate_dim, input_shape=(len(vocabulary_inv), embedding_dim))(input)
lstm = LSTM(intermediate_dim, return_sequences=False)(lstm)

z = Dense(intermediate_dim)(lstm)
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


model.summary()


# Train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('output/asm_best_model_lstm.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_val, y_val), callbacks=[es, mc], verbose=2)
model.save('output/model_lstm.h5')

# Evaluate
model = load_model('output/model_lstm.h5')
# model.evaluate(x_test, y_test, verbose=1)

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

    tot_bgn = np.sum(y == 0)
    tot_mal = np.sum(y == 1)

    tpr = 0
    far = 0
    for k,v in enumerate(y):
        # print(k, y[k], v)
        if v == 1 and y_preds[k] == 1:
            tpr += 1
        if v == 0 and y_preds[k] == 1:
            far += 1
    print('tpr', tpr, 'tot_mal', tot_mal, tpr/tot_mal)
    print('far', far, 'tot_bgn', tot_bgn, far/tot_bgn)

x = np.concatenate((x_train, x_val))
y = np.concatenate((y_train, y_val))
print('* Train result:')
pred(x, y)
print('* Test result:')
pred(x_test, y_test)