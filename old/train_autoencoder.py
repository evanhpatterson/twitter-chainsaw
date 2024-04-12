import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from old.autoencoder import create_autoencoder
import numpy as np
import random


def one_hot_encode_sequence(tokenizer: Tokenizer, sequence, max_length):
    encoded = np.zeros((max_length, tokenizer.num_words), dtype=np.float32)
    for i, word_index in enumerate(sequence):
        if i >= max_length:
            break
        encoded[i, word_index] = 1.0
    return encoded


vocab_size = 1000

split_fract = 0.9

raw_data = pd.read_csv("data/example_data.csv")
raw_data = list(raw_data['content'])


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(raw_data)

sequences = tokenizer.texts_to_sequences(raw_data)

max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)


output_sequences = [one_hot_encode_sequence(tokenizer=tokenizer, sequence=seq, max_length=max_length) for seq in padded_sequences]
output_sequences = np.array(output_sequences, dtype=np.float32)

# get permutation array for shuffle
permutation_arr = [i for i in range(len(padded_sequences))]

# shuffle permutation array
random.shuffle(permutation_arr)

# shuffle lists
padded_sequences = padded_sequences[permutation_arr]
output_sequences = output_sequences[permutation_arr]

# get cutoff point
cutoff_point = int(len(padded_sequences) * split_fract)

train_x = padded_sequences[:cutoff_point]
train_y = output_sequences[:cutoff_point]

test_x = padded_sequences[cutoff_point:]
test_y = output_sequences[cutoff_point:]


autoencoder, encoder, decoder = create_autoencoder(vocab_size=vocab_size, max_length=max_length, latent_dim=16)

autoencoder.summary()

autoencoder.fit(train_x, train_y, epochs=2, validation_data=(test_x, test_y))

