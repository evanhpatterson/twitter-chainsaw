import numpy as np
from gen_example_tweets import gen_example_tweets
from gen_example_stock_data import gen_stock_data
from wordprop.word_indexer import generate_word_dicts
from wordprop.internal_lexicon import load_internal_lexicon
from wordprop.encode_words import one_hot_encode_words
from wordprop.synsets_similarity import word_similarity_synsets
from wordprop.glove_similarity import word_similarity_glove_cos, word_similarity_glove_euc
from wordprop.create_word_matrix import create_word_matrix


if __name__=="__main__":
    # generate toy stock data
    stock_data = gen_stock_data("2019-01-01", "2019-06-01")

    # generate toy tweets
    tweet_data = gen_example_tweets(100, "2019-01-01 00:00:00", "2019-06-01 23:59:59")

    # concatenate tweets
    words_set = set((' '.join(tweet_data["text"])).split(sep=' '))
    unique_words = list(words_set)

    # get input word indices
    input_word_index, input_index_word = generate_word_dicts(unique_words)

    # get internal lexicon word indices
    internal_lexicon = load_internal_lexicon()
    output_word_index, output_index_word = generate_word_dicts(internal_lexicon)

    # encode to model inputs
    model_inputs = [
        one_hot_encode_words(
            words_list=tweet.split(sep=' '),
            input_word_index=input_word_index,
            max_size=128
            ) for tweet in tweet_data["text"]]
    
    model_inputs = np.array(model_inputs, dtype=np.float32)

    # generate the matrix representing similarities
    mat1 = create_word_matrix(
        unique_words, input_word_index,
        internal_lexicon, output_word_index,
        word_similarity_func=word_similarity_synsets)


    # generate the matrix representing similarities
    mat2 = create_word_matrix(
        unique_words, input_word_index,
        internal_lexicon, output_word_index,
        word_similarity_func=word_similarity_glove_cos)


    # generate the matrix representing similarities
    mat3 = create_word_matrix(
        unique_words, input_word_index,
        internal_lexicon, output_word_index,
        word_similarity_func=word_similarity_glove_euc)
    
    mat3 -= np.mean(mat3)
    mat3 /= np.std(mat3)
