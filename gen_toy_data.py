import numpy as np
from toy_data_generation.gen_example_tweets import gen_example_tweets
from toy_data_generation.gen_example_stock_data import gen_stock_data
from wordprop.word_indexer import generate_word_dicts
from wordprop.internal_lexicon import load_internal_lexicon
from wordprop.synsets_similarity import word_similarity_synsets
from wordprop.glove_similarity import word_similarity_glove_cos, word_similarity_glove_euc
from encode_data import encode_data
from wordprop.create_word_matrix import create_word_matrix
from wordprop.create_word_model import create_word_model
from wordprop.create_stocks_model import create_stocks_model
from keras.models import Sequential


if __name__=="__main__":
    start_date = "2019-01-01"
    end_date = "2019-06-01"

    # generate toy stock data
    stock_data = gen_stock_data(start_date=start_date, end_date=end_date)

    # generate toy tweets
    tweet_data = gen_example_tweets(10000, "2019-01-01 00:00:00", "2019-06-01 23:59:59")

    # concatenate tweets
    words_set = set((' '.join(tweet_data["text"])).split(sep=' '))
    unique_words = list(words_set)

    # get input word indices
    input_word_index, input_index_word = generate_word_dicts(unique_words)

    # get internal lexicon word indices
    internal_lexicon = load_internal_lexicon()
    output_word_index, output_index_word = generate_word_dicts(internal_lexicon)

    data_x, data_y = encode_data(tweet_data=tweet_data, stock_data=stock_data)

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
    
    stocks_model = create_stocks_model()

    model = Sequential()
    model.add(word_model)
    model.add(stocks_model)

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
