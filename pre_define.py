import json
from keras.datasets import imdb
import data_helpers
from w2v import train_word2vec
import numpy as np
np.random.seed(0)
from keras.preprocessing import sequence




# ---------------------- Parameters section -------------------
#
load_processed = True

test_pos_txt_path = "./data/none_.txt"
test_neg_txt_path = "./data/game_Linh_.txt"
# test_pos_txt_path = "./data/malware_test.txt"
# test_neg_txt_path = "./data/benign_test.txt"
train_pos_txt_path = "./data/malware_train.txt"
train_neg_txt_path = "./data/benign_train.txt"


# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "local_dir"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 10
filter_sizes = (3, 5)
num_filters = 4
# dropout_prob = (0.5, 0.8)
dropout_prob = (0.2, 0.3)
hidden_dims = 8

# Prepossessing parameters
sequence_length = 200
max_words = 1000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

#
# ---------------------- Parameters end -----------------------




def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_words, start_char=None, oov_char=None, index_from=None)
        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_val = sequence.pad_sequences(x_val, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        # save np.load
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

        x, y, x_test, y_test, vocabulary, vocabulary_inv_list = data_helpers.load_data(train_pos_txt_path, train_neg_txt_path, test_pos_txt_path, test_neg_txt_path, sequence_length=sequence_length)
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
if not load_processed:
    x_train, y_train, x_val, y_val, x_test, y_test, vocabulary_inv = load_data(data_source)

    np.save('data_prepared/x_train.npy', x_train)
    np.save('data_prepared/x_val.npy', x_val)
    np.save('data_prepared/y_train.npy', y_train)
    np.save('data_prepared/y_val.npy', y_val)
    with open('data_prepared/vocabulary_inv.json', 'w') as f:
        json.dump(vocabulary_inv, f)
    if test_pos_txt_path.split('/')[-1] == 'malware_test.txt':
        np.save('data_prepared/x_test.npy', x_test)
        np.save('data_prepared/y_test.npy', y_test)
    else: # none_bgn
        np.save('data/x_test__none_bgn.npy', x_test)
        np.save('data/y_test__none_bgn.npy', y_test)
else:
    # Load processed
    x_train = np.load('data_prepared/x_train.npy')
    y_train = np.load('data_prepared/y_train.npy')
    x_val = np.load('data_prepared/x_val.npy')
    y_val = np.load('data_prepared/y_val.npy')
    if test_pos_txt_path.split('/')[-1] == 'malware_test.txt':
        x_test = np.load('data/x_test.npy')
        y_test = np.load('data/y_test.npy')
    else:
        x_test = np.load('data/x_test__none_bgn.npy')
        y_test = np.load('data/y_test__none_bgn.npy')
    with open('data_prepared/vocabulary_inv.json', 'r') as f:
        vocabulary_inv = json.load(f)







print('sequence_length', sequence_length)
print('x_val.shape', x_val.shape)
if sequence_length != x_val.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_val.shape[1]

print("x_train shape:", x_train.shape)
print("x_val shape:", x_val.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_val)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_val = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_val])
        print("x_train static shape:", x_train.shape)
        print("x_val static shape:", x_val.shape)
elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")
