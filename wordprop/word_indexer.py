
def generate_word_dicts(words_list: list):
    '''
        Word dicts are generated that map words to indices and indices to words.
    '''
    word_index = dict()
    index_word = dict()
    for i, word in enumerate(words_list):
        word_index[word] = i
        index_word[i] = word
    return word_index, index_word

if __name__=="__main__":

    # create toy data
    input_words = ["water", "gun", "car", "distressed", "mad", "coin", "happy"]
    output_words = ["happy", "sad", "angry", "upset"]

    # generate dictionaries
    input_word_index, input_index_word = generate_word_dicts(input_words)
    output_word_index, output_index_word = generate_word_dicts(output_words)
