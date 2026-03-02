
# import required libraies

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, brentq


####################################### EXOGENOUS PARAMETERS

class Params():

    def __init__(self):

        self.K = 450        # fixed order cost
        self.gamma = 1      # discount factor
        self.b = 7/3        # backorder cost
        self.h = 1          # holding cost
        self.c = 0          # variable cost


####################################### LABELLING

def compute_bounds(D: np.array, K: float, gamma: float, b: float, h: float):

    """ Computes bounds for (s,S) - values, cf. Veinott & Wagner (1965).
    
    Parameters
    ----------
    D : (T,N) array_like
        Demand values for T periods, N samples per period
    K : real number
        fixed ordering cost
    gamma : real number
        discount factor
    b : real number
        backorder cost
    h : real number
        inventory holding cost

    Returns
    ----------
    bounds : tuple (low, high)
        for all t=0, ..., T-1: low<=s_t, S_t<=high
     """

    low, high = np.inf, -np.inf
    nv_quant = b/(b+h) # newsvendor quantile as initial guess for S_low

    for t in range(D.shape[0]):
        def one_period_cost(y):
            return np.mean( h * np.maximum(0, y-D[t, :]) + b * np.maximum(0, D[t, :]-y))
        S_low = np.quantile(D[t, :], nv_quant, interpolation='higher')
        cost_S_low = one_period_cost(S_low)
        high_t = S_low
        while one_period_cost(high_t) < cost_S_low + gamma*K:
            high_t += 1
        high = max(high, high_t)
        low_t = S_low
        while one_period_cost(low_t) < cost_S_low + K:
            low_t -= 1
        low = min(low, low_t)
    low, high = int(np.floor(low)), int(np.ceil(high))

    return low, high


def assemble_Gt(t: int, D_t: np.array, c: float, b: float, h: float, s_tp1: float, G_tp1_SK: float, G_tp1: float, last_period: int, y: np.array, gamma: float):

    """ Returns the function G_t as a callable, exactly evaluated at the grid points y and otherwise interpolated. Cf. Ban (2020) for details.
    
    Parameters
    ----------
    t : int
        Current period. Required to check if current is last period, as that requires a different G function.
    D_t : (1,N) array_like
        N demand samples in period t.
    c : real number
        variable ordering cost
    b : real number
        backorder cost
    h : real number
        inventory holding cost
    s_tp1 : real number
        Reorder point of next period; s_{t+1}
    G_tp1_SK : real number
        G function of next period evaluated at next_periods up-to-order level added to fixed ordering cost, pre-evaluated to reduce number of function calls.
        G_{t+1}(S_{t+1})+K
    G_tp1 : callable
        G function of next period; G_{t+1}
    last_period : int
        Last period of the horizon for which to compute (s_t,S_t)-values.
    y : (n, 1), array_like
        Grid on which to approximate G_t. The finer (larger n), the more accurate the approximation. Outside this grid, G is extrapolated linearly.
        Make sure (s_t, S_t) are within [y[0], y[-1]]. Cf. compute_bounds.
    gamma : real number
        discount factor
    

    Returns
    ----------
    G_t : callable
    """
    
    precomp = D_t-y # precomputed as this quantity is required multiple times
    G = c*y + b*np.maximum(precomp, 0) + h*np.maximum(-precomp, 0) # G in case t is the last period (there are no future periods)
    if t < last_period: # if t is not the last period, add future cost
        mask = (precomp+s_tp1>=0)
        G += gamma * ( c*(precomp) + G_tp1_SK*mask + G_tp1(-precomp)*(~mask) )
    return interp1d(y.flatten(), np.average(G, axis=1), assume_sorted=True, kind='cubic', fill_value='extrapolate') # function value at y[i] is the average over G[i, j], j=1, ..., N


def discrete_min_kconvex(G: callable, S_upper: int, K: float, step: int):

    """ Exact (s,S)-algorithm for discrete demand, cf. Bollapragada & Morton (1999).
        Extended by a step size parameter to trade off accuracy vs. runtime.
    
    Parameters
    ----------
    G : callable
        A
    S_upper : int
        Upper bound on S
    K : real number
        fixed ordering cost
    step : int
        Step size

    Returns
    ----------
    s : int
        Reorder point s
    S : int
        Up-to-order level s
    current_fv : real number
        G evaluated at 
    current_min : real number
        Minimum of G.
    nfev: int
        Number of function evaluations
    """
    
    # initialize values
    S, y, current_min = S_upper, S_upper, G(S_upper)
    current_fv = current_min
    nfev=1

    # Termination criterion exploits G's K-convexity in two ways:
    # 1. If current function value exceeds minimum by more than K, we have definitely passed the minimum and can stop
    # 2. The reorder point equals s<S such that G(s)=G(S)+K
    while current_fv < current_min + K: 
        y -= step # move leftwards
        current_fv = G(y) # re-evaluate at new guess
        if current_fv < current_min: # if function value at new S-guess is lower, update S estimate and minimum function value
            S = y
            current_min = current_fv
        nfev += 1 # update function evaluation counter
    s = y # reorder point is current function argument, as we now have S=argmin(G), G(s)>=G(S)+K
    
    return s, S, current_fv, current_min, nfev


def compute_labels(D: np.array, gamma: float, K: float, c: float, b: float, h: float, bounds: tuple, num: int=20000, step_initguess: int=5):
    
    """ 
    Computes the optimal (s,S)-labels.

    Parameters
    ----------
    D : (T, N) array_like
        Demand
    gamma : real number
        Discount factor
    K : real number
        Fixed order costs
    c : real number
        Variable order cost
    b : real number
        Backorder cost
    h : real number
        holding cost
    bounds : tuple (low, high)
        Must satisfy low<=s_t, S_t<=high for all t=0, ..., T-1
    num : int, optional
        Number of interpolation points on the interval [low, high]. Default=20000
    step_initguess : int, optional
    Step size to find initial guess for S and lower bound on s using discrete algorithm. Default=5.

    Returns
    ----------
    s_ls: list
        optimal reorder points (s_0, s_1, ..., s_{T-1})
    S_ls: list
        optimal up-to-order leves (S_0, S_1, ..., S_{T-1})
    """

    # initialize variables
    low, high = bounds
    y_interpol = np.linspace(low, high, num).reshape(-1, 1)
    s_tp1, G_tp1_SK, G_tp1 = 0, 0, 0
    s_ls, S_ls = [], []
    last_period = D.shape[0]-1
    
    # iterate through periods (from last to first) and compute period-wise optimal policy parameters
    for t in reversed(range(last_period+1)):
           
        # get period-wise G function
        Gt = assemble_Gt(t=t, D_t=D[[t], :], c=c, b=b, h=h, s_tp1=s_tp1, G_tp1_SK=G_tp1_SK, G_tp1=G_tp1, last_period=last_period, y=y_interpol, gamma=gamma)
        
        # get (s,S) estimates
        s_lower, S0, _, _, _ = discrete_min_kconvex(G=Gt, S_upper=high, K=K, step=step_initguess)

        # refine estimates:
        # S is function minimum
        res = minimize(fun=Gt, x0=S0, method='Nelder-Mead')
        S, G_tp1_SK = res.x[0], res.fun

        # s can be found with a line search (such that G(s)==G(S)+K)
        G_tp1_SK += K
        def root_fun(y):
            return Gt(y)-G_tp1_SK
        s_tp1 = brentq(f=root_fun, a=s_lower, b=S)
        
        # record period-wise values and G function for next period
        G_tp1 = Gt
        s_ls.append(s_tp1)
        S_ls.append(S)
    
    return list(reversed(s_ls)), list(reversed(S_ls)) # return in order (s_0, s_1, ..., s_{T-1})