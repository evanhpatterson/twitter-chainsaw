import numpy as np

def load_glove_model(glove_file):
    print("Loading Glove Model")
    f = open(glove_file,'r')
    glove_model = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = [float(value) for value in split_line[1:]]
        glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded")
    return glove_model

glove_path = 'glove.6B/glove.6B.50d.txt'
#glove_path = 'glove.6B/glove.6B.100d.txt'
#glove_path = 'glove.6B/glove.6B.200d.txt'
#glove_path = 'glove.6B/glove.6B.300d.txt'

glove_model = load_glove_model(glove_path)


def cosine_similarity(vec1, vec2):
    '''
        Calculate the cosine similarity between the two vectors.
    '''
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def word_similarity_glove(word1, word2):
    '''
        Compute the similarity between the word representations.
    '''
    # Get the vectors for each word from the glove_model
    vec1 = np.array(glove_model.get(word1))
    vec2 = np.array(glove_model.get(word2))

    if (vec1.dtype == np.float64) and (vec2.dtype == np.float64) and (vec1.shape == vec2.shape):
        cos_similarity = cosine_similarity(vec1, vec2)
        euc_similarity = np.linalg.norm(vec1 - vec2)
        euc_similarity = np.ones_like(euc_similarity) - euc_similarity
        return cos_similarity, euc_similarity
    else:
        return 0.0, float('-inf')

def word_similarity_glove_cos(word1, word2):
    cos_similarity, _ = word_similarity_glove(word1, word2)
    return cos_similarity

def word_similarity_glove_euc(word1, word2):
    _, euc_similarity = word_similarity_glove(word1, word2)
    return euc_similarity
