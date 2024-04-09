import numpy as np
from create_word_matrix import create_word_matrix
from word_indexer import generate_word_dicts
from create_nnet import create_model_from_word_matrix, create_word_to_stocks_model
from keras.models import Sequential
from keras.layers import Dense, Flatten


def evaluate_similarity_function(input_words, output_words, similarity_func, training_data, validation_data):
    '''
        See how well this one similarity function does at initializing weights.
    '''
    # generate dictionaries
    input_word_index, _ = generate_word_dicts(input_words)
    output_word_index, _ = generate_word_dicts(output_words)

    # use similarity metric to generate a similarity matrix
    mat = create_word_matrix(
        input_words=input_words,
        input_words_dict=input_word_index,
        output_words=output_words,
        output_words_dict=output_word_index,
        similarity_func=similarity_func)
    
    # create a neural net, initialize first layer with mat
    word_model = create_model_from_word_matrix(mat=mat, input_words=input_words, output_words=output_words)

    # make model for predicting stocks using words
    stocks_model = create_word_to_stocks_model()

    # combine models
    model = Sequential()
    model.add(word_model)
    model.add(stocks_model)

    # compile with adam
    model.compile(optimizer='adam', loss='mean_squared_error')

    # get training data
    training_data_x, training_data_y = training_data

    # train the model
    model.fit(training_data_x, training_data_y, epochs=100, validation_data=validation_data)


def test_similarity_functions(input_words, output_words, similarity_func_list, training_data, validation_data):
    '''
        Test each similarity metric in the list.
    '''
    for name, func in similarity_func_list:
        print(f"Testing {name}...")
        evaluate_similarity_function(
            input_words=input_words,
            output_words=output_words,
            similarity_func=func,
            training_data=training_data,
            validation_data=validation_data)
        print("\n\n")
