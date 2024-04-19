import numpy as np

def calc_direction_accuracy(pred, actual):
    '''
        Find the proportion of predictions are in
        the same direction as the actual values.
    '''
    components_equal = (np.sign(pred) == np.sign(actual))

    total = np.sum(components_equal)

    return total / components_equal.size
