from keras.models import Model
from keras.layers import Input, LSTM, Conv1D, TimeDistributed


def set_word_matrix(model, word_mat):
    weights = model.get_weights()

    weights[0][0, :word_mat.shape[0], :word_mat.shape[1]] = word_mat

    model.set_weights(weights)

def create_word_model(word_mat, num_unique_words, internal_lexicon_size, lstm1_size, lstm2_size, extra_words=0):
    
    input_layer = Input(shape=(None, None, num_unique_words))

    # this is what it will look like when we change this to an embedding layer
    #embedding_layer = TimeDistributed(Embedding(input_dim=num_unique_words, output_dim=internal_lexicon_size+extra_words))(input_layer)

    conv1d = TimeDistributed(
        Conv1D(
            filters=internal_lexicon_size+extra_words,
            kernel_size=1,
            activation='tanh'))(input_layer)
    
    lstm1 = TimeDistributed(LSTM(lstm1_size))(conv1d)

    lstm2 = LSTM(lstm2_size)(lstm1)

    model = Model(inputs=input_layer, outputs=lstm2)

    model.compile(optimizer='adam', loss='mean_squared_error')

    set_word_matrix(model=model, word_mat=word_mat)

    return model


if __name__=="__main__":
    from word_indexer import generate_word_dicts
    from create_word_matrix import create_word_matrix
    from glove_similarity import word_similarity_glove_cos

    # create toy data
    input_words = ["water", "gun", "car", "distressed", "mad", "coin", "happy"]
    output_words = ["happy", "sad", "upset"]

    # generate dictionaries
    input_word_index, input_index_word = generate_word_dicts(input_words)
    output_word_index, output_index_word = generate_word_dicts(output_words)

    # create the word matrix
    word_mat = create_word_matrix(
        input_words=input_words,
        input_words_dict=input_word_index,
        output_words=output_words,
        output_words_dict=output_word_index,
        similarity_func=word_similarity_glove_cos)
    
    print("word mat:")
    print(word_mat.T)
    print()

    model = create_word_model(
        word_mat=word_mat.T,
        num_unique_words=len(input_words),
        internal_lexicon_size=len(output_words),
        lstm1_size=32,
        lstm2_size=16, extra_words=1)
    
    print("model weights:")
    print(model.weights[0])
    print()

    print(model.input_shape)
    model.summary()
