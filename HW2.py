#Exercise 0
def github() -> str:
    """
    Returns the associated github link for this assignment
    """

    return "https://github.com/Julienmjohnson/ECON_481_HW_2/blob/main/HW2.py"




import numpy as np
import scipy as sp

#Exercise 1

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

#Exercise 2

def likelihood_fun(beta: np.array, y: np.array, X:np.array) -> float:
    """
    Returns the negative likelihood of seeing results y from the observation X with their coefficient estimates beta
    """
    y_pred = np.sum(np.hstack((np.ones(X.shape[0]).reshape(-1,1), X)) * beta, axis = 1).reshape((1000,1))
    
    likelihood = -0.5 * np.sum(np.log(2*np.pi) + np.square(y - y_pred))
    
    return -likelihood

def estimate_mle(y: np.array, X: np.array) -> np.array:
    """
    Returns an array of the MLE estimators for a data set
    """

    results = sp.optimize.minimize(
    fun=likelihood_fun, # the objective function
    x0= np.array([0.,1.,1.,1.]), # starting guess for coefficients
    args=(y,X), 
    method = 'Nelder-Mead' 
    )
    
    return results.x.reshape((4,1))    

#Exercise 3
def OLS_error_fun(beta: np.array, y: np.array, X:np.array) -> float:
    """
    Returns the square of the error term which is the difference between actual results y and the observation X with their coefficient estimations beta.
    """
    error = y - np.matmul(np.hstack((np.ones(X.shape[0]).reshape(-1,1), X)),beta.reshape(4,1))
   
    return np.dot(error.T,error)[0][0]

def estimate_ols(y: np.array, X: np.array) -> np.array:
    """
    Returns a 4x1 array of the OLS estimators for a data set
    """
    results = sp.optimize.minimize(
    fun=OLS_error_fun, # the objective function
    x0= np.array([0.,1.,1.,1.]), # starting guess
    args=(y,X), # additional parameters passed to neg_ll
    method = 'Nelder-Mead' # optionally pick an algorithm
    )

    return results.x.reshape((4,1))