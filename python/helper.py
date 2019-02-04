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
    f_s, t_s, s = spectrogram(wf, fs=fs)
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


### Patch-foraging theory ###
def cumulative_reward(t_p, R_0, r_0, tau):
    return r_0 * tau * (1.0 - np.exp(-t_p / tau)) + R_0
    
def get_optimal_values(t_t, R_0, r_0, tau):
    # Minimum travel time: R_0 / r_0
    if t_t < (R_0 / r_0):
        print('Travel time is less than minimum. Ignoring R_0.')
        R_0 = 0.0
    
    # Solve non-linear equation for residence time
    F = lambda x: (r_0 * np.exp(-x/tau) * (t_t + x + tau)) - (r_0 * tau) - R_0
    #t_p = scipy.optimize.broyden1(F, -100) # negative solution
    t_p_opt = broyden1(F, 10*tau) # positive solution
    
    # Calculate total harvested reward for optimal residence time
    r_opt = cumulative_reward(t_p_opt, R_0, r_0, tau)
    
    return t_p_opt, r_opt


### Utility functions ###
def _check_list(names):
    """
    Changes variable to list if not already an instance of one.
    """
    if not isinstance(names, list):
        names = [names]
    return names

def in_interval(t, t1, t2):
    gt_t1 = (t[np.newaxis, :] > t1[:, np.newaxis])
    lt_t2 = (t[np.newaxis, :] < t2[:, np.newaxis])
    
    return np.sum(np.logical_and(gt_t1, lt_t2).astype(np.int32), axis=0)