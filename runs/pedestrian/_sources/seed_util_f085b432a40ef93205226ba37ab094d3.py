import numpy as np

def retrieve_random_state(random_seed):
    """ Returns a random state if the seed is not a random state.
    """
    if not random_seed:
        r = np.random.RandomState(None)
    elif type(random_seed) == np.random.RandomState:
        r = random_seed
    else:
        r = np.random.RandomState(random_seed)
        
    return r