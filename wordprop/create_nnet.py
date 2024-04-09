from keras.models import Sequential
from keras.layers import Dense, Conv1D, Input, Flatten

def create_model_from_word_matrix(mat, input_words, output_words) -> Sequential:
    # create the model with the first input and output layers
    model = Sequential()
    model.add(Input(shape=(None, len(input_words))))
    model.add(Conv1D(len(output_words), kernel_size=1))

    # set the weights using the matrix
    weights = model.get_weights()
    weights[0][0] = mat.T
    model.set_weights(weights)

    return model

def create_word_to_stocks_model():

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model
