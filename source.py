
import numpy as np
from scipy.interpolate import interp1d

def compute_bounds(D: np.array, K: float, b: float, h: float, c: float, gamma: float):

    """ Computes bounds for (s,S) - values, cf. Veinott & Wagner (1965).
    
    Parameters
    ----------
    D : (T,N) array_like
        Demand values for T periods, N samples per period
    K : real number
        fixed ordering cost
    b : real number
        penalty cost
    h : real number
        inventory holding cost
    c : real number
        proportional ordering cost
    gamma : real number
        discount factor

    Returns
    ----------
    bounds : tuple (low, high)
        such that for all t=0, ..., T-1: low<=s_t<=S_t<=high
     """

    low, high = np.inf, -np.inf
    nv_quant = (b-(1-gamma)*c)/(b+h) # Veinott & Wagner, p. 537

    for t in range(D.shape[0]):

        def one_period_cost(y): # Veinott & Wagner, p. 527
            return (1-gamma)*c*y + np.mean( h * np.maximum(0, y-D[t, :]) + b * np.maximum(0, D[t, :]-y))
        
        S_low = np.quantile(D[t, :], nv_quant) # Veinott & Wagner, p. 537

        # Find upper bound; Veinott & Wagner, eqn. (21)
        cost_S_low = one_period_cost(S_low)
        high_t = S_low
        while one_period_cost(high_t) < cost_S_low + gamma*K:
            high_t += 1
        high_t = max(S_low, high_t-1)
        high = max(high, high_t)

        # Find lower bound; Veinott & Wagner, eqn. (22)
        low_t = S_low
        while one_period_cost(low_t) <= cost_S_low + K:
            low_t -= 1
        low = min(low, low_t+1)

    low, high = int(np.floor(low)), int(np.ceil(high))
    return low, high


def assemble_Gt(t: int, D_t: np.array, c: float, b: float, h: float, s_tp1: float, G_tp1_SK: float, G_tp1: float, last_period: int, y: np.array, gamma: float):

    """ Returns the function G_t as a callable, evaluated at the grid points y and otherwise interpolated. See Ban (2020) for details.
    
    Parameters
    ----------
    t : int
        Current period. Required to check if current is last period, as that requires a different G function.
    D_t : array of shape (1,N)
        N demand samples in period t.
    c : real number
        proportional ordering cost
    b : real number
        penalty cost
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
    y : array of shape (n, 1)
        Grid on which to approximate G_t. The finer (i.e, the larger n), the more accurate the approximation. Outside this grid, G is extrapolated linearly.
        Make sure (s_t, S_t) are within [y[0].item(), y[-1].item()]. See compute_bounds to derive limits.
    gamma : real number
        discount factor
    

    Returns
    ----------
    G_t : callable
    """
    
    precomp = D_t-y # precomputed as this quantity is required multiple times
    G = c*y + b*np.maximum(precomp, 0) + h*np.maximum(-precomp, 0) # G in case t is the last period (there are no future periods)
    if t < last_period: # if t is not the last period, add future cost
        mask = precomp+s_tp1>=0
        G += gamma * ( c*(precomp) + G_tp1_SK*mask + G_tp1(-precomp)*(~mask) )
    return interp1d(y.flatten(), np.average(G, axis=1), assume_sorted=True, fill_value='extrapolate') # function value at y[i] is the average over G[i, j], j=1, ..., N


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
    
    return s, S