import numpy as np

def one_hot_encode_words(text_str: str, input_word_index, max_size):
    '''
        Takes a list of words, outputs one-hot encodings of the words.
    '''
    encoded = np.zeros(shape=(max_size, len(input_word_index)), dtype=float)
    words_list = text_str.split(sep=' ')
    for i, word in enumerate(words_list):
        if word in input_word_index:
            j = input_word_index[word]
            encoded[i, j] = 1.0
    return encoded

def convert_to_ints(text_str: str, input_word_index, max_size):
    '''
        Takes a list of words, outputs a list of indices corresponding to words.
    '''
    encoded = np.zeros(max_size, dtype=int)
    words_list = text_str.split()

    for i, word in enumerate(words_list):
        if word in input_word_index:
            encoded[i] = input_word_index[word]

    return encoded
