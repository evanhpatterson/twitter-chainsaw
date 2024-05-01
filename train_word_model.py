import numpy as np
import pandas as pd
from wordprop.word_indexer import generate_word_dicts
from wordprop.internal_lexicon import load_internal_lexicon
from wordprop.create_word_matrix import create_word_matrix
from wordprop.create_word_model import create_word_model
from wordprop.synsets_similarity import word_similarity_synsets
from wordprop.glove_similarity import word_similarity_glove_cos, word_similarity_glove_euc
from keras.models import Sequential
from keras.layers import Dense
from preprocessing import preprocess_data


def train_word_model(data_x, data_y, input_word_index, unique_words):
    '''
        Build and train the initial model that uses words to make predictions.
    '''

    # get internal lexicon word indices
    internal_lexicon = load_internal_lexicon()
    output_word_index, _ = generate_word_dicts(internal_lexicon)

    # get the word matrix
    word_mat = create_word_matrix(
        input_words=unique_words,
        input_words_dict=input_word_index,
        output_words=internal_lexicon,
        output_words_dict=output_word_index,
        similarity_func=word_similarity_glove_cos)

    word_model = create_word_model(
        word_mat=word_mat.T,
        num_unique_words=len(unique_words),
        internal_lexicon_size=len(internal_lexicon),
        lstm1_size=128,
        lstm2_size=64,
        extra_words=0)

    model = Sequential()
    model.add(word_model)
    model.add(Dense(6))

    model.compile(optimizer='adam', loss='mse')

    model.fit(data_x, data_y, epochs=20)

    return word_model

if __name__=="__main__":

    tweet_fpath = "data/final_preprocessed_tweets.csv"

    stock_fpath = "data/NVDA-delta.csv"

    word_model = train_word_model(*preprocess_data(tweet_fpath=tweet_fpath, stock_fpath=stock_fpath))
