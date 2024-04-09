import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

def word_similarity_synsets(word1, word2):
    '''
        Return a float from 0 to 1 that tells you how similar the words are.
    '''

    # Get synsets for both words
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    max_similarity = 0

    # Iterate through all combinations of synsets to find the maximum similarity
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity

    return max_similarity
