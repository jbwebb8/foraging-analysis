import numpy as np
from scipy import signal
from scipy.optimize import broyden1, root
from scipy.optimize.nonlin import NoConvergence
import warnings

### Waveform analysis ###
def med_filt(x, n=3, ignore_nan=True):
    if ignore_nan:
        med_func = np.nanmedian
    else:
        med_func = np.median

    X = np.zeros([n, len(x)])
    for i in range(n):
        X[i] = np.roll(x, -n//2 + i)
    
    x_filt = med_func(X, axis=0)
    for i in range(n//2):
        x_filt[i] = med_func(X[(n//2 - i):, i])
        x_filt[-(i+1)] = med_func(X[:(n//2 + i), -(i+1)])
   
    return x_filt

def lowpass_butter_filter(s, fs, fc, order=5):
    """Apply low-pass butter filter to signal."""
    nyq = 0.5 * fs
    fc = fc / nyq
    b, a = signal.butter(order, fc, btype='low')
    y = signal.lfilter(b, a, s)
    
    return y

def smooth_waveform_variance(wf, fs, med_filter_size=30, butter_filter_fc=10):
    f_s, t_s, s = signal.spectrogram(wf, fs=fs)
    #s_var = np.sum((s - np.mean(s, axis=0, keepdims=True))**2, axis=0)
    s_var = np.median(np.abs(np.diff(s, axis=0)), axis=0)
    fs_s = fs * (len(s_var)/len(wf))
    s_var_smooth = med_filt(s_var, n=med_filter_size)
    s_var_smooth = lowpass_butter_filter(s_var_smooth, fs_s, butter_filter_fc)

    return fs_s, t_s, s_var_smooth

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
        # Corner case 1: last segment unreliably logged
        idx_last = min(len(dt_patch_1), len(dt_patch_2))

        # Corner case 2: first segment unreliably logged if too short
        if min(dt_patch_1[0], dt_patch_2[0]) < 0.5:
            dt_patch_1 = dt_patch_1[1:]
            dt_patch_2 = dt_patch_2[1:]
            idx_last -= 1

        # Corner case 3: only one patch, dropped above
        if ((dt_patch_1.size == 0 or dt_patch_2.size == 0)
            or (abs(dt_patch_1.shape[0] - dt_patch_2.shape[0]) > 1)):
            return False
        else:
            return np.isclose(dt_patch_1[:idx_last], 
                            dt_patch_2[:idx_last],
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
                       multipatch=False):
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
                                            max_value=max_value)
                                        

def _get_optimal_values_one_type(*, t_p=None, 
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
    
    # Convert to numpy arrays for matrix handling
    loc = [_to_array(kwarg) if kwarg is not None else None 
           for kwarg in loc]
    shape = ()
    for p in loc:
        if p is None:
            continue
        elif p.ndim == 1:
            shape += p.shape
        else:
            raise SyntaxError('Parameter rank must not exceed 1 (1D array)'
                              + ' but is {}.'.format(p.ndim))
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