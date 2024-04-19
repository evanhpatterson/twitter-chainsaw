import pandas as pd

def get_unique_words(tweet_data: pd.DataFrame):
    '''
        Generate a list of all unique words from the tweets dataset.
    '''
    # get a list of all words
    concatenated_words = (' '.join(tweet_data["text"])).split()

    # get all unique words
    words_set = set(concatenated_words)

    # turn the set of unique words into a list
    unique_words = list(words_set)

    return unique_words
