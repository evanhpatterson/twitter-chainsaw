import numpy as np
import pandas as pd
from wordprop.create_word_model import create_word_model

def train_word_model(data_x, data_y, params):
    '''
        Build and train the initial model that uses words to make predictions.
    '''
    
    model = create_word_model(*params)

    model.fit(data_x, data_y, epochs=20)

    return model
