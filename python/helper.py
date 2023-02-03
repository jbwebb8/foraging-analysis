import numpy as np
from scipy import signal, special
from scipy.optimize import broyden1, root, brentq
from scipy.optimize.nonlin import NoConvergence
import math
import warnings
import copy
import util
import itertools

### Waveform analysis ###
def med_filt(x, n=3, ignore_nan=True):
    # Determine handling of nan values
    if ignore_nan:
        med_func = np.nanmedian
    else:
        med_func = np.median

    return simple_filter(x, med_func, n)

def ma_filt(x, n=3, ignore_nan=True):
    # Determine handling of nan values
    if ignore_nan:
        ma_func = np.nanmean
    else:
        ma_func = np.mean

    return simple_filter(x, ma_func, n)

def simple_filter(x, func, n=3):
    # Group columns to which to apply function.
    X = np.zeros([n, len(x)])
    for i in range(n):
        X[i] = np.roll(x, -(n//2) + i)
    
    # Apply function to each column and correct end effects.
    x_filt = func(X, axis=0)
    for i in range(n//2):
        x_filt[i] = func(X[:(n//2 + i + 1), i])
        x_filt[-(i+1)] = func(X[(n//2 - i):, -(i+1)])
   
    return x_filt

def lowpass_butter_filter(s, fs, fc, order=5):
    """Apply low-pass butter filter to signal."""
    nyq = 0.5 * fs
    fc = fc / nyq
    b, a = signal.butter(order, fc, btype='low')
    y = signal.lfilter(b, a, s)
    
    return y

def smooth_waveform_variance(wf, fs, 
                             med_filter_size=30, 
                             butter_filter_fc=10):
    f_s, t_s, s = signal.spectrogram(wf, fs=fs)
    #s_var = np.sum((s - np.mean(s, axis=0, keepdims=True))**2, axis=0)
    s_var = np.median(np.abs(np.diff(s, axis=0)), axis=0)
    fs_s = fs * (len(s_var)/len(wf))
    s_var_smooth = med_filt(s_var, n=med_filter_size)
    s_var_smooth = lowpass_butter_filter(s_var_smooth, fs_s, butter_filter_fc)

    return fs_s, t_s, s_var_smooth

def smooth_waveform_power_ratio(wf, fs, 
                                fc=1000, 
                                med_filter_size=30,
                                med_filter_unit='samples', 
                                butter_filter_fc=10):
    """
    Computes the ratio of signal power concentrated below fc.
    """
    f_s, t_s, s = signal.spectrogram(wf, fs=fs)
    idx = np.sum(f_s < fc)
    r = np.sum(s[:idx, :], axis=0)/np.sum(s, axis=0)
    fs_s = fs * (len(r)/len(wf))
    if med_filter_size is not None:
        if med_filter_unit.lower() == 'time':
            med_filter_size = int(med_filter_size*fs_s)
        elif med_filter_unit.lower() != 'samples':
            raise ValueError('Unrecognized unit \'{}\'.'.format(med_filter_unit))
        r = med_filt(r, n=med_filter_size)
    if butter_filter_fc is not None:
        r = lowpass_butter_filter(r, fs_s, butter_filter_fc)

    return fs_s, t_s, r

def find_threshold(s, n_bins=25):
    """
    Finds threshold between two values for noisy binary function.
    Note: Fitting to two Gaussians often ignored the pink noise peak
    since it was proportionally much smaller.
    """
    # Create histogram of values
    hist, bin_edges = np.histogram(s, bins=n_bins)
    
    # Find local peaks
    peaks = []
    idx_peaks = []
    if hist[0] > hist[1]:
        peaks.append(hist[0])
        idx_peaks.append(0)
    for j in range(1, len(hist)-1):
        if hist[j] > hist[j-1] and hist[j] > hist[j+1]:
            peaks.append(hist[j])
            idx_peaks.append(j)
    if hist[-1] > hist[-2]:
        peaks.append(hist[-1])
        idx_peaks.append(-1)

    # Find values associated with greatest two local peaks
    idx_sort = np.argsort(np.asarray(peaks))
    idx_peaks = np.asarray(idx_peaks)[idx_sort[-2:]]
    bin_width = bin_edges[1] - bin_edges[0]
    x = bin_edges + 0.5*bin_width
    x_peaks = x[idx_peaks]

    # Return threshold halfway between two peaks
    return np.mean(x_peaks)

def compare_patch_times(dt_patch_1, dt_patch_2, atol=1.0, rtol=0.02):
        """
        Compare patch durations from two sources 
        (e.g. sound waveform analysis vs. logged data)
        
        Note: When comparing to logged data, the number of patches alone
        is not sufficient to check for the correct handling of the sound 
        waveform. Comparing times directly is not only more robust, but 
        if the session ended in a patch, it is not clear when that patch
        duration is logged, meaning the number of patches can differ by 
        one even with correct waveform analysis.
        """
        # Test for empty arrays
        if len(dt_patch_1) == 0 and len(dt_patch_2) == 0:
            return True
        elif len(dt_patch_1) == 0 or len(dt_patch_2) == 0:
            return False

        # # Corner case 1: last segment unreliably logged
        # idx_last = min(len(dt_patch_1), len(dt_patch_2))

        # # Corner case 2: first segment unreliably logged if too short
        # if min(dt_patch_1[0], dt_patch_2[0]) < 1.5:
        #     dt_patch_1 = dt_patch_1[1:]
        #     dt_patch_2 = dt_patch_2[1:]
        #     idx_last -= 1

        # # Corner case 3: only one patch, dropped above
        # if ((dt_patch_1.size == 0 or dt_patch_2.size == 0)
        #     or (abs(dt_patch_1.shape[0] - dt_patch_2.shape[0]) > 1)):
        #     return False
        # else:
        #     return np.isclose(dt_patch_1[:idx_last], 
        #                     dt_patch_2[:idx_last],
        #                     atol=atol, rtol=rtol).all()

        offset = len(dt_patch_1) - len(dt_patch_2)
        n = max(len(dt_patch_1), len(dt_patch_2))
        if abs(offset) > 2: # corner cases should only be first and/or last patch
            return False
        elif offset > 0:
            return any([np.isclose(dt_patch_1[i:n-(offset-i)], 
                                   dt_patch_2[:],
                                   atol=atol, rtol=rtol).all()
                        for i in range(offset+1)])
        elif offset < 0:
            return any([np.isclose(dt_patch_1[:], 
                                   dt_patch_2[i:n-(abs(offset)-i)],
                                   atol=atol, rtol=rtol).all()
                        for i in range(abs(offset)+1)])
        else:
            return np.isclose(dt_patch_1, 
                              dt_patch_2,
                              atol=atol, rtol=rtol).all()



### Patch-foraging theory ###
def _to_array(a, max_len=None):
    if isinstance(a, np.ndarray): # broadcast if needed
        return a
    elif isinstance(a, (int, float)): # assume single number
        return np.asarray([a])
    else:
        raise ValueError('Unable to convert type {} to ndarray.'
                         .format(type(a)))

def _broadcast_to_shape(a, shape, axis=None):
    """Broadcasts array to specified shape by matching dimensions of axes."""
    def to_tuple(name, param):
        if isinstance(param, list):
            return tuple(param)
        elif isinstance(param, int):
            return (param,)
        else:
            raise SyntaxError('{} must be tuple of int(s) but is type {}.'
                              .format(name, type(shape)))
    
    def raise_mismatch(axis, n):
        msg = 'Array axis {} of dimension {} does not match axis in shape parameter.' \
              .format(axis, n)
        raise ValueError(msg)
    
    # Convert parameters to proper format
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if not isinstance(shape, tuple):
        shape = to_tuple('shape', shape)
    
    s = [1]*len(shape) # rank of tensor shape
    
    # Match specified dimensions
    if axis is not None: 
        if not isinstance(axis, tuple):
            axis = to_tuple('axis', axis)
        if len(axis) != len(a.shape):
            if len(axis) == 1:
                axis = axis*len(a.shape)
            else:
                raise SyntaxError('Rank of input tensor and axis parameter must match.')
        for ax, n in zip(axis, a.shape):
            if shape[ax] != n:
                raise ValueError('Shape and axis parameters are not compatible.')
            else:
                s[ax] = n
    
    # Find first instance of matching dimensions
    else: 
        for i, n in enumerate(a.shape):
            if n > 1:
                try:
                    idx = shape.index(n)
                    while s[idx] == n:
                        idx = shape.index(n, idx+1)
                    s[idx] = n
                except ValueError:
                    raise_mismatch(i, n)
        
    return a.reshape(s)*np.ones(shape)

def random_slice(X, axis=0):
    """Return random slice of numpy array"""

    assert isinstance(X, np.ndarray)
    
    if X.ndim < 1:
        return X
    elif X.ndim < 2:
        return X[np.random.randint(len(X))]
    else:
        idx = ()
        if not isinstance(axis, tuple):
            if isinstance(axis, list):
                axis = tuple(axis)
            else:
                axis = (axis,)
        
        start = 0
        for i in axis:
            for j in range(start, i):
                idx += (np.random.randint(X.shape[j]),)
            idx += (slice(None),)
            start = i + 1
        for j in range(start, len(X.shape)):
            idx += (np.random.randint(X.shape[j]),)

        return X[idx]

def _broyden1(f, x_init, args=(), max_iter=10000, min_value=-np.inf, max_value=np.inf):
    soln = root(f, x_init, method='broyden1', args=args, options={'maxiter': max_iter})
    if min_value is None:
        min_value = -np.inf
    if max_value is None:
        max_value = np.inf
    if not soln.success:
        raise NoConvergence(soln.message)
    elif (soln.x < min_value).any() or (soln.x > max_value).any():
        raise OverflowError('Solution outside of allowed range.')
    return soln.x


def _solve_mvt_equation(*, F_log, F_exp, x_init, verbose=False, **kwargs):
    with np.errstate(divide='raise', invalid='raise'):
        try:
            return _broyden1(F_log, x_init, **kwargs) # positive solution
        except (NoConvergence, FloatingPointError, OverflowError):
            if verbose:
                print('Logarithmic function failed. Attempting to solve exponential function...')
            return _broyden1(F_exp, x_init, **kwargs) # positive solution

def cumulative_reward(t_p, R_0, r_0, tau):
    return r_0 * tau * (1.0 - np.exp(-t_p / tau)) + R_0
    
def get_optimal_values(*, t_p=None, 
                       t_t=None, 
                       R_0=None, 
                       r_0=None, 
                       tau=None,
                       return_solvable=False,
                       min_value=None,
                       max_value=None,
                       multipatch=False,
                       broadcast_shape=True):
    """
    Returns optimal value based on the marginal value theorem, based on values
    of other parameters.

    Args:
    - t_p: Patch residence time.
    - t_t: Travel time.
    - R_0: Initial reward volume.
    - r_0: Initial reward rate.
    - tau: Decay constant.
    - return_solvable: If True, return array of whether each point is solvable.
    - multipatch: If True, solve environment with multiple patch types.

    Returns:
    - opt: Optimal value of unknown parameter.
    - R_opt: Amount of cumulative reward at optimal leaving time.
    - is_solvable: Array where True means data point has solution.
    """
    if multipatch:
        return _get_optimal_values_multi_type(t_p=t_p, 
                                              t_t=t_t, 
                                              R_0=R_0, 
                                              r_0=r_0, 
                                              tau=tau,
                                              return_solvable=return_solvable,
                                              min_value=min_value,
                                              max_value=max_value)
    else:
        return _get_optimal_values_one_type(t_p=t_p, 
                                            t_t=t_t, 
                                            R_0=R_0, 
                                            r_0=r_0, 
                                            tau=tau,
                                            return_solvable=return_solvable,
                                            min_value=min_value,
                                            max_value=max_value,
                                            broadcast_shape=broadcast_shape)
                                        

def _get_optimal_values_one_type(*, t_p=None, 
                                 t_t=None, 
                                 R_0=None, 
                                 r_0=None, 
                                 tau=None,
                                 return_solvable=False,
                                 min_value=None,
                                 max_value=None,
                                 broadcast_shape=True):
    # Assert one unknown
    loc = [t_p, t_t, R_0, r_0, tau]
    n_unknown = len([kwarg for kwarg in loc if kwarg is None])
    if n_unknown < 1:
        raise SyntaxError('No unknowns.')
    elif n_unknown > 1:
        raise SyntaxError('Too many unknowns. Please specify all but one parameter.')
    
    # Convert to numpy arrays for matrix handling
    loc = [_to_array(kwarg) if kwarg is not None else None 
           for kwarg in loc]
    if broadcast_shape:
        shape = ()
        for p in loc:
            if p is None:
                continue
            elif p.ndim == 1:
                shape += p.shape
            else:
                raise SyntaxError('Parameter rank must not exceed 1 (1D array)'
                                + ' but is {}.'.format(p.ndim))
    else:
        # Ensure all shapes are either (1,) or (n,)
        n_max = max([len(kwarg) for kwarg in loc if kwarg is not None])
        assert all([len(kwarg) in [1, n_max] for kwarg in loc if kwarg is not None])
        shape = (n_max,)
    loc = [_broadcast_to_shape(kwarg, shape) if kwarg is not None else None 
            for kwarg in loc]
    [t_p, t_t, R_0, r_0, tau] = loc

    # Minimum travel time: R_0 / r_0
    is_solvable = np.ones(shape, dtype=np.bool)
    if (t_t is not None) and (R_0 is not None) and (r_0 is not None):
        if np.sum(t_t < (R_0 / r_0)) > 0:
            warnings.warn('Some data points exceed minimum travel time. R_0 set to zero.')
            idx = t_t < (R_0 / r_0)
            R_0[idx] = 0.0
            is_solvable[idx] = False
    
    # Solve equation based on unknown.
    # The trickiest part of solving the equation is managing the initial condition
    # for the non-linear solver. Having the wrong initial condition can lead to
    # overflow errors or a nonsense solution (negative or very large, since the
    # equations can have multiple solutions). We will handle these issues with
    # nested try-except statements, as well as iterating over different initial states.
    with np.errstate(divide='raise', invalid='raise'): # force to raise error to catch
        if t_p is None:
            depth = 0 # depth of indices to iterate over
            t_p = np.empty(shape) # initialize empty solution array
            solved = np.zeros(shape, dtype=np.bool) # track which elements have been solved
            while depth <= len(shape):
                try:
                    # Iterate over indices at depth of dimension
                    for idx in np.ndindex(shape[:depth]):
                        if not solved[idx].all():
                            # Solve non-linear equation for residence time (logarithmic form more numerically stable)
                            [t_t_, R_0_, r_0_, tau_] = [p[idx] for p in loc if p is not None ]
                            F_log = lambda x: (np.log(r_0_) - (x / tau_) + np.log(t_t_ + x + tau_) - np.log((r_0_ * tau_) + R_0_))
                            F_exp = lambda x: (r_0_ * np.exp(-x/tau_) * (t_t_ + x + tau_)) - (r_0_ * tau_) - R_0_
                            t_p[idx] = _solve_mvt_equation(F_log=F_log, F_exp=F_exp, x_init=tau_, min_value=min_value, max_value=max_value)
                            solved[idx] = True # update successful solutions
                    break # stop if found solution for all indices
                except (NoConvergence, FloatingPointError, OverflowError) as error:
                    # Catch numerical errors and try at deeper iteration
                    depth += 1
                    if depth > len(shape):
                        print([t_t_, R_0_, r_0_, tau_])   
                        raise NoConvergence('Failed to coverge to solution.')
                    else:
                        print('Solution failed. Attempting to solve at depth {} (of max {})...'
                              .format(depth, len(shape)))
            opt = t_p
        elif t_t is None:
            # Solve for travel time
            t_t = ((r_0 * tau) + R_0)/(r_0 * np.exp(-t_p/tau)) + (t_p - tau)
            opt = t_t
        elif R_0 is None:
            # Solve for initial reward
            R_0 = (r_0 * np.exp(-t_p/tau) * (t_t + t_p + tau)) - (r_0 * tau)
            opt = R_0
        elif r_0 is None:
            # Solve for initial rate of reward
            if R_0 == 0.0:
                print('Initial rate of return can be any positive number.')
            r_0 = R_0 / (np.exp(-t_p/tau) * (t_t + t_p + tau) - tau)
            opt = r_0
        elif tau is None:
            depth = 0 # depth of indices to iterate over
            tau = np.empty(shape) # initialize empty solution array
            solved = np.zeros(shape, dtype=np.bool) # track which elements have been solved
            while depth <= len(shape):    
                try:
                    # Iterate over indices at depth of dimension
                    for idx in np.ndindex(shape[:depth]):
                        if not solved[idx].all():
                            # Solve non-linear equation for decay rate (logarithmic form more numerically stable)
                            [t_p_, t_t_, R_0_, r_0_] = [p[idx] for p in loc if p is not None]
                            F_log = lambda x: (np.log(r_0_) - (t_p_ / x) + np.log(t_t_ + t_p_ + x) - np.log((r_0_ * x) + R_0_))
                            F_exp = lambda x: (r_0_ * np.exp(-t_p_/x) * (t_t_ + t_p_ + x)) - (r_0_ * x) - R_0_
                            if depth < len(shape):
                                # Solve multidimensional equation
                                try:
                                    tau[idx] = _solve_mvt_equation(F_log=F_log, 
                                                                F_exp=F_exp, 
                                                                x_init=t_t_, 
                                                                min_value=min_value, 
                                                                max_value=max_value)
                                except (NoConvergence, FloatingPointError, OverflowError) as error:
                                    tau[idx] = _solve_mvt_equation(F_log=F_log, 
                                                                F_exp=F_exp, 
                                                                x_init=t_p_, 
                                                                min_value=min_value, 
                                                                max_value=max_value)
                            else:
                                # Solve individual points. Here, we can cover a reasonable space
                                # of initial conditions (in 1D) to better find a solution.
                                x_init = np.geomspace(0.1, max(t_p_, t_t_), num=10)
                                for x in x_init:
                                    try:
                                        tau[idx] = _solve_mvt_equation(F_log=F_log, 
                                                                    F_exp=F_exp, 
                                                                    x_init=x, 
                                                                    min_value=min_value, 
                                                                    max_value=max_value)
                                        break # stop if found solution for given x_init
                                    except (NoConvergence, FloatingPointError, OverflowError) as error:
                                        continue # try again with next x_init
                            solved[idx] = True # update successful solutions
                    break # stop if found solution for all indices
                except (NoConvergence, FloatingPointError, OverflowError) as error:
                    # Catch numerical errors and try at deeper iteration
                    depth += 1
                    if depth > len(shape):
                        raise NoConvergence('Failed to coverge to solution.')
                    else:
                        print('Solution failed. Attempting to solve at depth {} (of max {})...'
                            .format(depth, len(shape)))
            opt = tau
    
    # Calculate total harvested reward for optimal residence time
    R_opt = cumulative_reward(t_p, R_0, r_0, tau)
    
    # Return number if only one value queried
    if isinstance(opt, np.ndarray) and not isinstance(R_opt, np.ndarray):
        opt = float(opt)
    
    if return_solvable:
        return opt, R_opt, is_solvable
    else:
        return opt, R_opt

def _get_optimal_values_multi_type(*, t_p=None, 
                                   t_t=None, 
                                   R_0=None, 
                                   r_0=None, 
                                   tau=None,
                                   return_solvable=False,
                                   min_value=None,
                                   max_value=None):
    # Assert one unknown
    loc = [t_p, t_t, R_0, r_0, tau]
    n_unknown = len([kwarg for kwarg in loc if kwarg is None])
    if n_unknown < 1:
        raise SyntaxError('No unknowns.')
    elif n_unknown > 1:
        raise SyntaxError('Too many unknowns. Please specify all but one parameter.')

    # Determine number of patches
    loc = [_to_array(kwarg) if kwarg is not None else None 
           for kwarg in loc]
    shape = None
    for p in loc:
        if p is None:
            continue
        elif shape is None:
            shape = p.shape
        elif shape != p.shape:
            raise SyntaxError('All inputs must have the same shape for multipatch'
                              + ' calculation ({} does not match {}).'.format(p.shape, shape))
    if len(shape) == 1:
        shape = (1, shape[0])
    elif len(shape) > 2:
        raise SyntaxError('Parameter rank must be 1 or 2 only but is {}.'.format(len(shape)))
    [t_p, t_t, R_0, r_0, tau] = loc
    M = shape[0] # number of environments to solve
    N = shape[1] # number of patch types

    # Create reward expressions
    r = lambda t_p, tau, r_0: r_0*np.exp(-t_p/tau)
    R = lambda t_p, tau, r_0, R_0: r_0*tau*(1.0 - np.exp(-t_p/tau)) + R_0

    # Define system of non-linear equations:
    # r_i(t_p_i)*(sum(t_t) + sum(t_p)) - sum(R(t_p) - s*t_t) = 0
    def F(x, idx):
        # Pre-process parameters for system of non-linear equations
        nonlocal t_p, t_t, r_0, R_0, tau # grab argument values from above
        X = np.zeros((5,) + (1,)*(x.ndim == 1) + x.shape) # all parameters
        for i, p in enumerate([t_p, t_t, r_0, R_0, tau]):
            if p is None:
                X[i] = x
            else:
                X[i] = p[idx]
        
        # Pre-compute sums before loop
        sum_R = np.sum(np.vstack([R(X[0,:,i], X[4,:,i], X[2,:,i], X[3,:,i]) for i in range(N)]).T, axis=1)
        sum_T = np.sum(X[0,:,:], axis=1) + np.sum(X[1,:,:], axis=1)
        
        # Loop through each patch type
        y = np.zeros((1,)*(x.ndim == 1) + x.shape)
        for i in range(N):
            y[:, i]= r(X[0,:,i], X[4,:,i], X[2,:,i])*sum_T - sum_R

        return y

    # Because we created a generic function to optimize, we can use the same loop for
    # any unknown variable. The only difference is the initial value for the function
    # solver.
    if t_p is None:
        x_init = tau
    elif t_t is None:
        # Without search cost, only the sum of travel times matters to the solution.
        # Thus we will make the initial value the same for each row to give a solution
        # with equal travel times for all patches.
        x_init = np.mean(t_p, axis=1, keepdims=True)*np.ones(shape)
    elif tau is None:
        x_init = t_p
    with np.errstate(divide='raise', invalid='raise'): # force to raise error to catch
        depth = 0 # depth of indices to iterate over
        opt = np.empty(shape) # initialize empty solution array
        solved = np.zeros(shape, dtype=np.bool) # track which elements have been solved
        while depth < len(shape):
            try:
                # Iterate over indices at depth of dimension
                for idx in np.ndindex(shape[:depth]):
                    if not solved[idx].all():
                        # Solve system of non-linear equations
                        opt[idx] = _broyden1(F, x_init[idx],
                                             args=(idx,),
                                             min_value=min_value, 
                                             max_value=max_value)
                        solved[idx] = True # update successful solutions
                break # stop if found solution for all indices
            except (NoConvergence, FloatingPointError, OverflowError) as error:
                # Catch numerical errors and try at deeper iteration
                depth += 1
                if depth == len(shape):
                    # We cannot iterate element by element because multipatch 
                    # case requires solving system as whole.  
                    raise NoConvergence('Failed to coverge to solution.')
                else:
                    print('Solution failed. Attempting to solve at depth {} (of max {})...'
                            .format(depth, len(shape)))
    
    # Calculate total harvested reward for optimal residence time
    params = [p if p is not None else opt for p in [t_p, R_0, r_0, tau]]
    R_opt = cumulative_reward(*params)
    
    # Return number if only one value queried
    if isinstance(opt, np.ndarray) and not isinstance(R_opt, np.ndarray):
        opt = float(opt)
    
    if return_solvable:
        return opt, R_opt, solved
    else:
        return opt, R_opt

def estimate_parameters(t_k, t_bin,
                        T=None, 
                        model='full', 
                        history='equal', 
                        L=1,
                        p_lam0=(1,0),
                        p_tau=(1,0)):
    """
    Estimate parameters (lam_0, tau) of non-homogeneous Poisson process
    given event times t_k.
    
    Args:
    - t_k (list): List of numpy arrays, each of which is the event (reward)
        times for a given patch sequence.
    - t_bin (np.ndarray): Times at which to estimate the parameters. Must
        define interval [0.0, t], where t is less than or equal to the 
        last residence time.
    - T (float, np.ndarray): Patch residence times for the sequences given in
        t_k. If a single number, then it is assumed all residence times are equal.
    - model (str): Model type to estimate underlying parameters.
    - history (str): History of previous sequences to feed to model.
        - equal: For every time 0 < t < T in current patch, the model will 
            receive information up to time t for each of the previous patch
            sequences. This is useful for evaluating estimates for a given
            fixed residence time.
        - consecutive: For every time 0 < t < T in current patch, the model will
             receive information from all previous sequences and up to time t
             in current patch. This is simulating the observations an animal
             would see during an experimental session.
        
    Returns:
    - lam_0 (np.ndarray): Estimated lam_0 parameter for t_bin.
    - tau (np.ndarray): Estimated tau parameter for t_bin.
    """
    # Check parameters
    if history not in ['equal', 'consecutive']:
        raise ValueError('Unknown history \'{}\''.format(history))
    if not isinstance(t_k, list):
        raise SyntaxError('t_k must be a list of arrays.')
    if T is None:
        T = t_bin[-1:] # make numpy array
    elif not isinstance(T, np.ndarray):
        T = np.atleast_1d(T)
    if (T[1:] == T[0]).all():
        T = T[:1] # keep as numpy array
    elif len(T) > 1:
        assert len(T) == len(t_k)  
    if t_bin[0] != 0.0:
        raise ValueError('Time bins must start at 0.0')
    if t_bin[-2] > T[-1]:
        raise ValueError('Time bins must not exceed last residence time.')
    if len(p_lam0) != len(p_tau) != 2:
        raise SyntaxError('Parameter priors must be tuples of length 2.')
        
    # Drop last (current) residence time if using previous patch sequences.
    # If using only current sequence, variable T is cancelled out by (m-1)
    # terms in multisequence computations.
    if len(T) > 1:
        T = T[:-1]

    if model.lower() == 'full':
        return _full_process_MLE(t_k, t_bin, T, history)
    elif model.lower() == 'hidden':
        return _hidden_process_MLE(t_k, t_bin, T, history, L, p_lam0, p_tau)
    else:
        raise ValueError('Unknown model type \'{}\''.format(model))
        
def _full_process_MLE(t_k, t_bin, T, history):
    # NOTE: Function no longer maintained!
    warnings.warn('This function is no longer maintained and does not use priors.'
                  + ' Use _hidden_process_MLE with L=1 instead.',
                  category=DeprecationWarning)
            
    # Variables
    m = len(t_k)
    tau_MLE = np.zeros([len(t_bin)-1])
    lam_MLE = np.zeros([len(t_bin)-1])
    
    # Calculate MLE at each time point
    for i, t in enumerate(t_bin[1:]):
        if history == 'equal':
            # Calculate means of variables
            t_k_ = np.hstack(t_k)
            t_sum_mean = (1.0/m)*np.sum(t_k_[t_k_ <= t])
            K_mean = (1.0/m)*np.sum(t_k_ <= t)

            # Define function
            f = lambda tau: (K_mean*tau - t_sum_mean)*(np.exp(t/tau) - 1.0)/t - K_mean
            tau_init = t_sum_mean/K_mean

            # Find MLE of tau
            if t_sum_mean < 0.5*K_mean*t:
                tau_MLE[i] = brentq(f, tau_init, 1e6)
            else:
                tau_MLE[i] = math.inf

            # Find corresponding MLE of lambda
            if tau_MLE[i] < math.inf:
                lam_MLE[i] = (K_mean - t_sum_mean/tau_MLE[i])/t
            else:
                lam_MLE[i] = K_mean/t
        
        elif history == 'consecutive':
            # Calculate variables
            if m > 1:
                t_k_ = np.hstack(t_k[:-1]) # use all times from previous sequences
                K_ = len(t_k_)
                t_k_ = np.hstack([t_k_, t_k[-1][t_k[-1] <= t]]) # add times from current sequence
                K_m = np.sum(t_k[-1] <= t)
            else:
                t_k_ = t_k[0][t_k[0] <= t]
                K_ = 0
                K_m = len(t_k_)
            t_sum = np.sum(t_k_)
            K = K_ + K_m
            
            # If all previous residence times equal, then can simplify to sum.
            if len(T) == 1:
                # Define function
                f = lambda tau: ( (m - (m-1)*np.exp(-T/tau) - np.exp(-t/tau))
                                  /((m-1)*T*np.exp(-T/tau) + t*np.exp(-t/tau))
                                  *(K*tau - t_sum)
                                  - K
                                )
                tau_init = t_sum/K

                # Find MLE of tau
                if t_sum < 0.5*(K_*m*T + K_m*t):
                    tau_MLE[i] = brentq(f, tau_init, 1e6)
                else:
                    tau_MLE[i] = math.inf
                
                # Find corresponding MLE of lambda
                if tau_MLE[i] < math.inf:
                    lam_MLE[i] = ( (K*np.exp(-t/tau_MLE[i]))
                                   /(tau_MLE[i]*(m - (m-1)*np.exp(-T/tau_MLE[i]) - np.exp(-t/tau_MLE[i])))
                                 ) 
                else:
                    lam_MLE[i] = K/((m-1)*T + t)
        
            # Otherwise, must compute sum of exponential terms.
            else:
                # Define function
                f = lambda tau: ( (m - np.sum(np.exp(-T/tau)) - np.exp(-t/tau))
                                  /(np.sum(T*np.exp(-T/tau)) + t*np.exp(-t/tau))
                                  *(K*tau - t_sum)
                                  - K
                                )
                tau_init = t_sum/K

                # Find MLE of tau
                if t_sum < 0.5*(np.sum([len(t_k_m)*T_m for t_k_m, T_m in zip(t_k[:-1], T)])
                                + len(t_k[-1])*t):
                    try:
                        tau_MLE[i] = brentq(f, tau_init, 1e6)
                    except ValueError:
                        tau_MLE[i] = math.inf
                else:
                    tau_MLE[i] = math.inf

                # Find corresponding MLE of lambda
                if tau_MLE[i] < math.inf:
                    lam_MLE[i] = ( (K*np.exp(-t/tau_MLE[i]))
                                   /(tau_MLE[i]*(m - np.sum(np.exp(-T/tau_MLE[i])) - np.exp(-t/tau_MLE[i])))
                                 ) 
                else:
                    lam_MLE[i] = K/(np.sum(T) + t)
        
    return (lam_MLE, tau_MLE)

def _hidden_process_MLE(t_k, t_bin, T, history, L,
                        p_lam0=(1,0), p_tau=(1,0)):
    # Variables
    m = len(t_k)
    tau_MLE = np.zeros([len(t_bin)-1])
    lam_MLE = np.zeros([len(t_bin)-1])

    # Get priors on estimated parameters.
    alpha_1, beta_1 = p_lam0
    alpha_2, beta_2 = p_tau
    
    # Calculate MLE at each time point
    for i, t in enumerate(t_bin[1:]):
        if history == 'equal':
            # Calculate means of variables
            t_k_ = np.hstack(t_k)
            t_sum = np.sum(t_k_[t_k_ <= t])
            K = np.sum(np.hstack(t_k) <= t)

            # Define function over reward segments.
            def phi_k(tau):
                phi = lambda t1, t2: (t1*np.exp(-t1/tau) - t2*np.exp(-t2/tau))/(np.exp(-t1/tau) - np.exp(-t2/tau))
                phi_k = []
                for t_k_ in t_k:
                    t_k_t = np.insert(t_k_[t_k_ <= t], 0, 0.0)
                    phi_k.append(phi(t_k_t[:-1], t_k_t[1:]))
                return np.hstack(phi_k)

            # Set previous residence times to t.
            T = t
        
        elif history == 'consecutive':
            # Calculate variables
            if m > 1:
                t_k_ = np.hstack(t_k[:-1]) # use all times from previous sequences
                K_ = len(t_k_)
                t_k_ = np.hstack([t_k_, t_k[-1][t_k[-1] <= t]]) # add times from current sequence
                K_m = np.sum(t_k[-1] <= t)
            else:
                t_k_ = t_k[0][t_k[0] <= t]
                K_ = 0
                K_m = len(t_k_)
            t_sum = np.sum(t_k_)
            K = K_ + K_m
            
            # Define function over reward segments.
            def phi_k(tau):
                phi = lambda t1, t2: (t1*np.exp(-t1/tau) - t2*np.exp(-t2/tau))/(np.exp(-t1/tau) - np.exp(-t2/tau))
                phi_k = []
                for t_k_ in t_k[:-1]:
                    t_k_t = np.insert(t_k_, 0, 0.0)
                    phi_k.append(phi(t_k_t[:-1], t_k_t[1:]))
                t_k_t = np.insert(t_k[-1][t_k[-1] <= t], 0, 0.0)
                phi_k.append(phi(t_k_t[:-1], t_k_t[1:]))
                return np.hstack(phi_k)
            
            # Place dummy residence time for calculation.
            # EDIT: This leads to a bug with the term np.sum(np.exp(-T/tau))
            # incorrectly evaluated as 1 in code block with len(T) > 0.
            #if m == 1:
            #    T = np.zeros([1])
            
        # If all previous residence times equal, then can simplify to sum.
        if len(T) == 1:
            # Define function
            f = lambda tau: ( ((K+alpha_1-1)*tau - t_sum - (L-1)*np.sum(phi_k(tau))
                                - (alpha_2-1)*tau + beta_2*tau**2)
                                *(m - (m-1)*np.exp(-T/tau) - np.exp(-t/tau) + beta_1/tau)
                                /((m-1)*T*np.exp(-T/tau) + t*np.exp(-t/tau) + beta_1)
                                - (L*K + alpha_1 - 1)
                            )
            tau_init = 0.1

            # Find MLE of tau
            if K > 0: # avoid trivial equation
                try:
                    tau_MLE[i] = brentq(f, tau_init, 1e6)
                except ValueError:
                    tau_MLE[i] = math.inf
            else:
                tau_MLE[i] = math.inf

            # Find corresponding MLE of lambda
            if tau_MLE[i] < math.inf:
                lam_MLE[i] = ( ((L*K + alpha_1 - 1)*np.exp(-t/tau_MLE[i]))
                                /(tau_MLE[i]*(m - (m-1)*np.exp(-T/tau_MLE[i]) - np.exp(-t/tau_MLE[i])) + beta_1)
                                ) 
            else:
                lam_MLE[i] = (L*K + alpha_1 - 1)/((m-1)*T + t + beta_1)
        
        # Otherwise, must compute sum of exponential terms.
        else:
            # Define function
            f = lambda tau: ( ((K+alpha_1-1)*tau - t_sum - (L-1)*np.sum(phi_k(tau))
                                - (alpha_2-1)*tau + beta_2*tau**2)
                                *(m - np.sum(np.exp(-T/tau)) - np.exp(-t/tau) + beta_1/tau)
                                /(np.sum(T*np.exp(-T/tau)) + t*np.exp(-t/tau) + beta_1)
                                - (L*K + alpha_1 - 1)
                            )
            a = 0.1
            b_range = np.geomspace(0.5, 1e6, 50)

            # Find MLE of tau
            success = False
            if K > 0: # avoid trivial equation
                for b in b_range:
                    try:
                        tau_MLE[i] = brentq(f, a, b)
                        success = True
                        break
                    except ValueError:
                        continue
            if not success:
                tau_MLE[i] = math.inf

            # Find corresponding MLE of lambda
            if tau_MLE[i] < math.inf:
                lam_MLE[i] = ( ((L*K + alpha_1 - 1)*np.exp(-t/tau_MLE[i]))
                                /(tau_MLE[i]*(m - np.sum(np.exp(-T/tau_MLE[i])) - np.exp(-t/tau_MLE[i])) + beta_1)
                             ) 
            else:
                lam_MLE[i] = (L*K + alpha_1 - 1)/(np.sum(T) + t + beta_1)
        
    return (lam_MLE, tau_MLE)

def estimate_log_likelihood(t_k,
                            t_bin,
                            T=None,
                            lam=None,
                            tau=None,
                            model='full', 
                            history='equal', 
                            L=1,
                            p_lam0=(1,0),
                            p_tau=(1,0),
                            epsilon=0.1):
    # Check parameters
    if history not in ['equal', 'consecutive']:
        raise ValueError('Unknown history \'{}\''.format(history))
    if not isinstance(t_k, list):
        raise SyntaxError('t_k must be a list of arrays.')
    if T is None:
        T = t_bin[-1:] # make numpy array
    elif not isinstance(T, np.ndarray):
        T = np.atleast_1d(T)
    if (T[1:] == T[0]).all():
        T = T[:1] # keep as numpy array
    elif len(T) > 1:
        assert len(T) == len(t_k)  
    if t_bin[0] != 0.0:
        raise ValueError('Time bins must start at 0.0')
    if t_bin[-2] > T[-1]:
        raise ValueError('Time bins must not exceed last residence time.')
        
    # Drop last (current) residence time if using previous patch sequences.
    # If using only current sequence, variable T is cancelled out by (m-1)
    # terms in multisequence computations.
    if len(T) > 1:
        T = T[:-1]

    if model.lower() == 'full':
        return _full_process_LL(t_k, t_bin, T, lam, tau, history)
    elif model.lower() == 'hidden':
        return _hidden_process_LL(t_k, t_bin, T, lam, tau, history, L,
                                  p_lam0, p_tau, epsilon)
    else:
        raise ValueError('Unknown model type \'{}\''.format(model))

def _full_process_LL(t_k, t_bin, T, lam, tau, history):
    # NOTE: Function no longer maintained!
    warnings.warn('This function is no longer maintained and does not use priors.'
                  + ' Use _hidden_process_LL with L=1 instead.',
                  category=DeprecationWarning)

    # Variables
    m = len(t_k)
    logp = np.zeros([len(t_bin)-1])
    if len(T) == 1:
        T_prev = np.ones([len(t_k)-1])*T
    else:
        T_prev = T[:-1]
    
    # Define equation components
    def Lam_n(tau, lam_0): 
        if tau == np.inf:
            Lam_ = lambda t1, t2: lam_0*(t2 - t1)
        else:
            Lam_ = lambda t1, t2: -lam_0*tau*(np.exp(-t2/tau) - np.exp(-t1/tau)) 
        Lam_n = np.zeros(len(t_bin)-1)
        for i in range(len(t_bin)-1):
            Lam_n[i] = Lam_(t_bin[i], t_bin[i+1])
        return Lam_n
    
    # Calculate MLE at each time point
    for i, t in enumerate(t_bin[1:]):
        # Return negative infinity if estimated Poisson rate is zero.
        if lam[i] == 0.0:
            logp[i] = -np.inf
            continue

        # Calculate Lam_n for given tau
        lam_0 = lam[i]*np.exp(t/tau[i])
        Lam_n_ = Lam_n(tau[i], lam_0)

        if history == 'equal':
            # Filter event times
            t_k_ = [t_k_[t_k_ <= t] for t_k_ in t_k]
            t_bin_ = t_bin[t_bin <= t]
            idx = len(t_bin_)

            # Compute history-independent component
            if tau[i] == np.inf:
                ll = -m*lam_0*t
            else:
                ll = -m*lam_0*tau[i]*(1.0 - np.exp(-t/tau[i]))
            
            # Compute history-dependent component
            for t_k_i in t_k_:
                k_n = util.in_interval(t_k_i, t_bin_[:-1], t_bin_[1:], query='interval')
                ll += (np.sum(k_n*np.log(Lam_n_[:idx-1]))
                       - np.sum(np.log(special.factorial(k_n))))
        
        elif history == 'consecutive':
            # Vectorize previous and current patch times
            T_ = np.append(T_prev, t)

            # Compute history-independent component
            if tau[i] == np.inf:
                ll = -lam_0*np.sum(T_)
            else:
                ll = np.sum(-lam_0*tau[i]*(1.0 - np.exp(-T_/tau[i])))

            # Compute history-dependent component
            for t_k_, T_m in zip(t_k, T_):
                t_bin_ = t_bin[t_bin <= T_m]
                idx = len(t_bin_)
                k_n = util.in_interval(t_k_, t_bin_[:-1], t_bin_[1:], query='interval')
                ll += (np.sum(k_n*np.log(Lam_n_[:idx-1]))
                       - np.sum(np.log(special.factorial(k_n))))

        # Save log-likelihood
        logp[i] = ll

    return logp
        
def _hidden_process_LL(t_k, t_bin, T, lam, tau, history, L,
                       p_lam0=(1,0),
                       p_tau=(1,0),
                       epsilon=0.1):
    # Variables
    m = len(t_k) # number of sequences observed
    logp = np.zeros([len(t_bin)-1])
    if len(T) == 1:
        T_prev = np.ones([len(t_k)-1])*T
    else:
        T_prev = T[:-1]

    # Define function components 
    def Lam_n(tau, lam_0, t_k): 
        if tau == np.inf:
            Lam_ = lambda t1, t2: lam_0*(t2 - t1)
        else:
            Lam_ = lambda t1, t2: -lam_0*tau*(np.exp(-t2/tau) - np.exp(-t1/tau)) 
        Lam_n = []
        for t_k_ in t_k:
            if len(t_k_) > 0:
                t_k_ = np.insert(t_k_, 0, 0.0)
                Lam_n.append(Lam_(t_k_[:-1], t_k_[1:]))
        if len(Lam_n) > 0:
            return np.hstack(Lam_n)
        else:
            return np.nan
    
    # Calculate log-likelihood at each time point
    for i, t in enumerate(t_bin[1:]):
        # Return negative infinity if estimated Poisson rate is zero.
        if lam[i] == 0.0:
            logp[i] = -np.inf
            continue

        # Calculate initial lambda for given tau, t
        lam_0 = lam[i]*np.exp(t/tau[i])

        if history == 'equal':
            # Filter event times
            t_k_ = [t_k_[t_k_ <= t] for t_k_ in t_k]

            # Compute history-independent component
            if tau[i] == np.inf:
                ll = -m*lam_0*t
            else:
                ll = -m*lam_0*tau[i]*(1.0 - np.exp(-t/tau[i]))
        
        elif history == 'consecutive':
            # Vectorize previous and current patch times
            T_ = np.append(T_prev, t)

            # Filter event times
            t_k_ = copy.deepcopy(t_k)
            t_k_[-1] = t_k_[-1][t_k_[-1] <= t]

            # Compute history-independent component
            if tau[i] == np.inf:
                ll = -lam_0*np.sum(T_)
            else:
                ll = np.sum(-lam_0*tau[i]*(1.0 - np.exp(-T_/tau[i])))

        # Get number of observed events
        K = 0 # total number of events observed
        t_sum = 0.0 # sum of observation times
        for t_k_i in t_k_:
            K += len(t_k_i)
            t_sum += np.sum(t_k_i)

        # Compute history-dependent component
        ll += (L-1)*np.sum(np.log(Lam_n(tau[i], lam_0, t_k_))) # Lambda(t_n-1, t_n)
        ll -= K*(np.log(special.factorial(L-1))) 
        ll += K*np.log(lam_0) - t_sum/tau[i] + K*np.log(epsilon) # lambda(t)

        # Compute prior components.
        for (alpha, beta), val in zip([p_lam0, p_tau], [lam_0, tau[i]]):
            if np.isfinite(val):
                ll += (alpha-1)*np.log(val) - beta*val # value-dependent components
            ll += alpha*np.log(beta) - np.log(special.gamma(alpha)) # constants
        
        # Save log-likelihood
        logp[i] = ll

    return logp

def calculate_parameters(t_k,
                         t_bin,
                         T=None,
                         lam=None,
                         tau=None,
                         model='full', 
                         history='equal', 
                         L=1,
                         p_lam0=(1,0),
                         p_tau=(1,0)):
    """Compute values of one parameter given values of the other."""
    # Check parameters
    if not isinstance(t_k, list):
        raise SyntaxError('t_k must be a list of arrays.')
    if T is None:
        T = t_bin[-1:] # make numpy array
    elif not isinstance(T, np.ndarray):
        T = np.atleast_1d(T)
    assert len(T) == len(t_k)  
    if t_bin[0] != 0.0:
        raise ValueError('Time bins must start at 0.0')
    if t_bin[-2] > T[-1]:
        raise ValueError('Time bins must not exceed last residence time.')

    # Check inputs
    if not (lam is None)^(tau is None):
        raise SyntaxError('Only one variable must be specified as None.')
    elif tau is not None:
        if not isinstance(tau, np.ndarray):
            tau = np.array([[tau]])
        elif tau.ndim == 1:
            tau = tau[:, np.newaxis] # broadcast time on axis 1
    elif lam is not None:
        if not isinstance(lam, np.ndarray):
            lam = np.array([[lam]])
        elif lam.ndim == 1:
            lam = lam[:, np.newaxis] # broadcast time on axis 1

    # Get priors on estimated parameters.
    alpha_1, beta_1 = p_lam0
    alpha_2, beta_2 = p_tau

    # Calculate number of events at each time step
    if model == 'full':
        L = 1
    elif model != 'hidden':
        raise ValueError('Unknown model {}.'.format(model))
    if history == 'full':
        K = np.sum(np.hstack(t_k)[:, np.newaxis] <= t_bin[np.newaxis, 1:], axis=0)
    elif history == 'consecutive':
        K = sum([len(t_k_) for t_k_ in t_k[:-1]])
        K += util.in_interval(t_k[-1], 
                              np.zeros([len(t_bin)-1]), 
                              t_bin[1:], 
                              query='interval')
        K = K[np.newaxis, :] # broadcast time on axis 1

    # Calculate unknown parameter values
    if tau is None:
        raise NotImplementedError('Solving for tau not yet implemented.')
    elif lam is None:
        # Create placeholder array
        lam = np.ones([tau.shape[0], t_bin.shape[0]-1])*np.nan
        t = t_bin[np.newaxis, 1:]
        T = T[:-1]
        m = len(t_k)

        # Handle finite values (non-homogeneous process)
        idx = (tau < np.inf)[:, 0]
        if len(T) > 0:
            exp_sum = np.sum(np.exp(-T[np.newaxis, :]/tau[idx]), axis=1, keepdims=True)
        else:
            exp_sum = 0.0
        lam[idx, :] = ( ((L*K + alpha_1 - 1)*np.exp(-t/tau[idx]))
                         /(tau[idx]*(m - exp_sum - np.exp(-t/tau[idx])) + beta_1)
                      ) 
        
        # Handle infinite values (homogeneous process)
        lam[~idx, :] = (L*K + alpha_1 - 1)/(np.sum(T) + t + beta_1)

        return lam


### Tree data structures ###
class Tree:
    
    def __init__(self, name, value):
        """
        Class for handling hierarchical data. One caveat to the traditional
        data structure is that the penultimate level simply has the values,
        rather than object pointers, of its children. This allows for faster
        subsampling in the generate_sample() method.
        """
        self.name = name
        self.value = value
        self.parent = None
        self.children = None
        self.penultimate = False
        
    def set_parent(self, parent):
        self.parent = parent
        
    def set_children(self, children):
        self.children = children
        
    def get_ancestry(self, order='descending'):
        """Returns list of parent Tree objects in specified order."""
        ancestry = []
        tree = self.parent
        while tree is not None:
            ancestry.append(tree)
            tree = tree.parent
        
        if order == 'ascending':
            return ancestry
        elif order == 'descending':
            return ancestry[::-1]
        else:
            raise ValueError('Unknown error {}.'.format(order))
    
    def get_index(self, order='descending'):
        """
        Returns values of node ancestry in specified order. Ignores
        parent values if None.
        """
        # Allow for values to be tuples.
        index = [p.value for p in self.get_ancestry(order=order) 
                 if p is not None]
        index = [val if isinstance(val, (tuple, list)) else [val]
                 for val in index]
        index = list(itertools.chain.from_iterable(index))
        
        if order == 'ascending':
            index.insert(0, self.value)
        elif order == 'descending':
            index.append(self.value)
        else:
            raise ValueError('Unknown error {}.'.format(order))
            
        return tuple(index)
   
    def generate_sample(self, seed=None, size=None, subsize=None, verbose=False):
        """
        Generates a sample of the leaf nodes by randomly sampling, with
        replacement, at each level of the tree. Done many times, this is 
        equivalent to hierarchical bootstrapping.

        Args:
        - seed: Random seed for random number generator.
        - size: Total size of sample. If none, returns first subsample,
            regardless of size.
        - subsize: Size of sampling at each level. If None, the sampling size is equal
            to the number of nodes in the original dataset at that level. If -1, then 
            the level is sampled in its entirety without replacement (i.e. original data).
        - verbose: If True, notify user if node has less than subsize samples.

        Returns:
        - sample: 1D array of sampled values.
        """
        # Check overall size.
        if size is None:
            size = 1 # generate single subsample

        # Check subsample size.
        if subsize is None:
            subsize = {}
        elif not isinstance(subsize, dict):
            subsize = {level: subsize for level in self.get_sublevels()}
        elif any([k not in self.get_sublevels() for k in subsize.keys()]):
            names = [k for k in subsize.keys() if k not in self.get_sublevels()]
            warnings.warn('Unknown level(s) {}.'.format(names),
                          category=RuntimeWarning)
        
        # Build sample until size requirement satisfied.
        sample = []
        rng = np.random.default_rng(seed=seed)
        while len(sample) < size:
            subsample = []
            self._generate_sample(self, rng, subsample, subsize, verbose)
            subsample = np.hstack(subsample)
            sample = np.hstack([sample, subsample])
            
        # Randomly trim last subsample to ensure equal size.
        if size > 1:
            diff = len(sample) - size
            idx = np.arange(len(sample) - len(subsample), len(sample))
            idx = rng.choice(idx, size=len(subsample)-diff, replace=False)
            idx = np.hstack([np.arange(len(sample) - len(subsample)), idx])
            sample = sample[idx]
            assert len(sample) == size
            
        return sample
    
    @staticmethod
    def _generate_sample(tree, rng, sample, subsize, verbose=False):
        """Recursive method for traversing tree by sampling with replacement."""
        # Create random sample of children.
        size = subsize.get(tree.get_next_level(), len(tree.children))
        if len(tree.children) < size and verbose:
            warnings.warn('Subsample size is larger than number of child nodes.',
                          category=RuntimeWarning)
        if size > 0 :
            idx = rng.choice(np.arange(len(tree.children)), 
                             size=size,
                             replace=True)
        elif size == -1:
            idx = np.arange(len(tree.children))
        else:
            raise ValueError(f'Value of {size} not allowed for size parameter.')
        
        if tree.penultimate:
            # If penultimate node, return sample of children values.
            return tree.children[idx]
        else:
            # Otherwise, return values from traversing tree 
            # at each sampled child node.
            for i in idx:
                s = Tree._generate_sample(tree.children[i], rng, sample, subsize, verbose)
                if s is not None:
                    sample.append(s)

    def get_next_level(self):
        if len(self.children) > 0:
            if self.penultimate:
                return 'leaf'
            else:
                name = self.children[0].name
                assert all([child.name ==  name for child in self.children])
                return name
        else:
            return None

    def get_sublevels(self):
        return self._get_sublevels([])

    def _get_sublevels(self, levels):
        if len(self.children) > 0:
            if self.penultimate:
                name = 'leaf'
            else:
                name = self.children[0].name
                assert all([child.name ==  name for child in self.children])

            if name not in levels:
               levels.append(name)

        if not self.penultimate:
            for child in self.children:
                child._get_sublevels(levels)
        
        return levels

    def get_level_counts(self):
        counts = self._get_level_counts({})
        return {k: np.array(v) for k, v in counts.items()}
    
    def _get_level_counts(self, counts):
        if len(self.children) > 0:
            if self.penultimate:
                name = 'leaf'
            else:
                name = self.children[0].name
                assert all([child.name ==  name for child in self.children])
            
            if name not in counts.keys():
                counts[name] = []
            counts[name].append(len(self.children))

        if not self.penultimate:
            for child in self.children:
                child._get_level_counts(counts)
            
        return counts


def make_tree(df, levels, init_value):
    """
    Recursive method for building tree structure from hierarchical data.
    Note that the dataframe must have multiindexing, which can be achieved
    by the following:

    df = df.set_index([index_1, index_2, ...])

    such that subsets can be located by:

    subset = df.loc[val_1, val_2, ...]

    The initial value provided pertains to the first level in levels.
    """
    return _make_tree(df, levels, 0, None, init_value)

def _make_tree(df, levels, depth, parent, value):
    # Handle leaves of tree
    if depth >= len(levels):
        return None
    elif depth == len(levels) - 1:
        return value
    else:
        # Create tree at current depth
        tree = Tree(levels[depth], value)
        tree.set_parent(parent)
    
    # Determine unique values of next hierarchical level.
    index = tree.get_index()
    level = levels[depth+1] 
    vals = sorted(df.loc[index].index.unique(level=level))

    # Create child for each unique value.
    children = []
    for val in vals:
        children.append(_make_tree(df, levels, depth+1, tree, val))
    
    # Set children.
    if depth == len(levels) - 2:
        # At penultimate node, simply set children equal to their values.
        tree.penultimate = True
        children = np.array(children)
    tree.set_children(children)
    
    return tree

def check_tree(tree, n):
    """Check that each node at the level above animals has at least n animals."""
    if len(tree.children) > 0:
        if tree.children[0].name == 'mouse_id':
            if len(tree.children) < n:
                warnings.warn('Node at {} only has {} animals.'
                              .format(tree.get_index(), len(tree.children)),
                              category=RuntimeWarning)
        else:
            for child in tree.children:
                check_tree(child, n)
        