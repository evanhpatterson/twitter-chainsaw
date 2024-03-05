'''
    Example data from IMDB,
    Use this to see if sentiment analysis is working.
'''
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# Parameters
vocab_size = 1000
max_length = 200 # Max length of each review
embedding_dim = 16 # Dimensionality of the embedding layer
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>" # Out-of-Vocabulary token


# Load IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# pad sequences
train_padded = pad_sequences(train_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_padded = pad_sequences(test_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)
