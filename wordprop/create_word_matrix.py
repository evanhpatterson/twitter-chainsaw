import numpy as np

def create_word_matrix(
        input_words: list, input_words_dict: dict,
        output_words: list, output_words_dict: dict, similarity_func):
    '''
        Generates a matrix where entry[i, j] represents the similarity between
        word[i] and word[j].
        The matrix performs a linear transformation on an input vector
        that outputs a vector representing how much that word or words
        represents the concepts associated with the output words.
    '''
    mat = np.zeros(shape=(len(output_words), len(input_words)))
    for iword in input_words:
        for oword in output_words:
            input_index = input_words_dict[iword]
            output_index = output_words_dict[oword]
            mat[output_index, input_index] = similarity_func(iword, oword)
    return mat
