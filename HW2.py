#Exercise 0
def github() -> str:
    """
    Returns the associated github link for this assignment
    """

    return "https://github.com/Julienmjohnson/ECON_481_HW_2/blob/main/HW2.py"


#Exercise 1

import numpy as np

def simulate_data(seed: int=481) -> tuple:
    """
    Returns 1000 simulated observations via the following data generating process mentioned in Website 1 as a tuple, 
    the first element being predicted output and the second being the array of observations for each element.
    The function takes a seed as an integer with the default being 481
    """
    rng = np.random.default_rng(seed)
    X = 0+2*rng.standard_normal(size=1000*3).reshape((1000,3))
    coef = np.array([5,3,2,6,1]).reshape((1,5))
    Y_unsummed = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X, rng.standard_normal(size=1000).reshape((1000,1)))) * coef
    Y_summed = np.sum(Y_unsummed, axis = 1).reshape((1000,1))

    return (Y_summed,X)