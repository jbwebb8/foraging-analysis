from helper import _check_list, in_interval

# Initial methods to make:
# - harvest rate
#   - overall
#   - per patch
# - patch residence and travel time
#   - experimental values
#   - optimal values
# - lick
#   - in vs. out of patch
#   - d` and bias
# - motor
#   - observed reward per patch

# These methods should not be repetitive and use only lower level
# methods that are abstracted in the Session class. In other words,
# they should be the leaves in a dependency tree; otherwise, they
# should be moved to a Session (sub)class method.

# TODO: have load_precomputed method to handle loading saved data
# to avoid having to recalculate all sessions to handle only new ones

def get_lick_decisions(sess, min_interval=None):
    """
    Returns timestamps for decisions to lick during the session. These are
    defined by any lick with a preceding inter-lick interval that is greater
    than the inter-lick interval for lick-bouts.

    Args: 
    - sess: Session instance

    Returns:
    - t_decision: DAQ timestamps associated with a licking decision.
    """
    # Get lick timestamps
    t_lick = sess.get_lick_times()
    dt_lick = np.diff(t_lick)

    # Get histogram of inter-lick intervals <= 1.0 seconds
    hist, bin_edges = np.histogram(dt_lick, range=[0.0, 1.0], bins=20)
    bin_width = bin_edges[1] - bin_edges[0] # 50 ms
    
    # Find max peak associated with lick-bout
    h_max = np.max(hist)
    idx_max = np.argmax(hist)

    # Set threshold as first bin significantly after peak
    thresh = bin_edges[idx_max + 1] + 0.5*bin_width
    for i in range(idx_max + 1, len(hist)):
        if hist[i] < (0.1 * h_max):
            thresh = bin_edges[i] + 0.5*bin_width
            break
    
    # Filter licks by time from previous lick
    idx_decision = np.insert(np.diff(t_lick) > thresh, 0, 0)
    t_decision = t_lick[idx_decision]

    return t_decision
    
def get_lick_stats(sess, per_patch=True):
    """
    Returns statistics for licking in vs. out of patch.

    Args:
    - sess: Session instance

    Returns:
    - stats: dictionary of lick statistics
        - n_total: total number of lick decisions
        - n_patch: number of lick decisions in patch(es)
        - n_interpatch: number of lick decisions in interpatch(es)
        - f_patch: lick decision rate in patch(es)
        - f_interpatch: lick decision rate in interpatch(es)
    """
    # Get timestamps associated with lick decisions
    t_decision = get_lick_decisions(sess)

    # Get timestamps associated with patch entry/exit
    t_patch = sess.get_patch_times()

    # Determine number of licks per patch/interpatch
    n_patch = in_interval(t_decision, 
                          t_patch[:, 0], 
                          t_patch[:, 1])
    n_interpatch = in_interval(t_decision, 
                               t_patch[:, 1], 
                               np.append(t_patch[1:, 0], sess.vars['t_total']))
    
    # Determine lick rates per patch/interpatch
    dt_patch = sess.get_patch_durations()
    dt_interpatch = sess.get_interpatch_durations()
    if per_patch:
        f_patch = n_patch / dt_patch
        f_interpatch = n_interpatch / dt_interpatch
    else:
        n_patch = np.sum(n_patch)
        n_interpatch = np.sum(n_interpatch)
        f_patch = n_patch / np.sum(dt_patch)
        f_interpatch = n_interpatch / np.sum(dt_interpatch)

    lick_stats = {'n_total': n_total,
                  'n_patch': n_patch,
                  'n_interpatch': n_interpatch,
                  'f_patch': f_patch,
                  'f_interpatch': f_interpatch}
    
    return lick_stats




    
