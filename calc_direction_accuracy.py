import numpy as np

def calc_direction_accuracy(pred, actual) -> float:
    '''
        Find the proportion of predictions are in
        the same direction as the actual values.
    '''
    components_equal = (np.sign(pred) == np.sign(actual))
    
    total = np.sum(components_equal)

    return total / components_equal.size

if __name__=="__main__":
    pred = np.random.randint(0, 4, (10, 5))
    actual = np.random.randint(0, 4, (10, 5))
    
    print(pred)
    print(actual)
    
    print("____ ____ ____ ____")
    
    print(calc_direction_accuracy(pred, actual))
