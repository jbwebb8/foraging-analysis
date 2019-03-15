import numpy as np
from scipy import signal
from scipy.optimize import broyden1
import warnings

### Waveform analysis ###
def med_filt(x, n=3):
    X = np.zeros([n, len(x)])
    for i in range(n):
        X[i] = np.roll(x, -n//2 + i)
    
    x_filt = np.median(X, axis=0)
    for i in range(n//2):
        x_filt[i] = np.median(X[(n//2 - i):, i])
        x_filt[-(i+1)] = np.median(X[:(n//2 + i), -(i+1)])
   
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

        return np.isclose(dt_patch_1[:idx_last], 
                          dt_patch_2[:idx_last],
                          atol=atol, rtol=rtol).all()


### Patch-foraging theory ###
def _to_array(a, max_len=None):
    if max_len is None:
        max_len = len(a)
    if isinstance(a, np.ndarray): # broadcast if needed
        return a * np.ones([max_len])
    else: # assume single number
        return np.asarray([a]) * np.ones([max_len])

def cumulative_reward(t_p, R_0, r_0, tau):
    return r_0 * tau * (1.0 - np.exp(-t_p / tau)) + R_0
    
def get_optimal_values(*, t_p=None, t_t=None, R_0=None, r_0=None, tau=None):
    """
    Returns optimal value based on the marginal value theorem, based on values
    of other parameters.

    Args:
    - t_p: Patch residence time.
    - t_t: Travel time.
    - R_0: Initial reward volume.
    - r_0: Initial reward rate.
    - tau: Decay constant.

    Returns:
    - opt: Optimal value of unknown parameter.
    - r_opt: Amount of cumulative reward at optimal leaving time.
    """

    # Assert one unknown
    loc = [t_p, t_t, R_0, r_0, tau]
    n_unknown = len([kwarg for kwarg in loc if kwarg is None])
    if n_unknown < 1:
        raise SyntaxError('No unknowns.')
    elif n_unknown > 1:
        raise SyntaxError('Too many unknowns. Please specify all but one parameter.')
    
    # Convert to numpy arrays for vector handling
    max_len = max([len(kwarg) if isinstance(kwarg, np.ndarray) else 1
                   for kwarg in loc])
    [t_p, t_t, R_0, r_0, tau] = [_to_array(kwarg, max_len) if kwarg is not None else None 
                                 for kwarg in loc]

    # Minimum travel time: R_0 / r_0
    if (t_t is not None) and (R_0 is not None) and (r_0 is not None):
        if np.sum(t_t < (R_0 / r_0)) > 0:
            warnings.warn('Some data points exceed minimum travel time. R_0 set to zero.')
            R_0[t_t < (R_0 / r_0)] = 0.0
    
    # Solve equation based on unknown
    if t_p is None:
        # Solve non-linear equation for residence time
        F = lambda x: (r_0 * np.exp(-x/tau) * (t_t + x + tau)) - (r_0 * tau) - R_0
        #t_p = scipy.optimize.broyden1(F, -100) # negative solution
        t_p = broyden1(F, 10*tau) # positive solution
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
        # Solve non-linear equation for decay rate
        F = lambda x: (r_0 * np.exp(-t_p/x) * (t_t + t_p + x)) - (r_0 * x) - R_0
        #tau = scipy.optimize.broyden1(F, -100) # negative solution
        tau = broyden1(F, t_p) # positive solution
        opt = tau
    
    # Calculate total harvested reward for optimal residence time
    r_opt = cumulative_reward(t_p, R_0, r_0, tau)
    
    # Return number if only one value queried
    if isinstance(opt, np.ndarray) and not isinstance(r_opt, np.ndarray):
        opt = float(opt)
    
    return opt, r_opt