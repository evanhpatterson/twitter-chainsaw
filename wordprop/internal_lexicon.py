from wordprop.word_indexer import generate_word_dicts

def load_internal_lexicon():
    return list(set(open('wordprop/internal_lexicon.txt', 'r').read().split(sep='\n')))
