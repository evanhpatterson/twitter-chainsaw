
def load_internal_lexicon():
    contents = open('wordprop/internal_lexicon.txt', 'r').read().replace('(', ' ').replace(')', ' ').lower()
    split_text = list(set(contents.split()))
    return [word for word in split_text if not (len(word) == 1)]
