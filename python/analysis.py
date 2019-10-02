from util import _check_list, in_interval
import numpy as np

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

    if min_interval is None:
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
        print('Found threshold of %.2f seconds.' % thresh)
    else:
        # Set threshold to specified interval
        thresh = min_interval

    # Filter licks by time from previous lick
    idx_decision = np.insert(np.diff(t_lick) > thresh, 0, True)
    t_decision = t_lick[idx_decision]

    return t_decision
    
def get_lick_stats(sess, per_patch=True, min_interval=None):
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
    t_decision = get_lick_decisions(sess, min_interval=min_interval)

    # Get timestamps associated with patch entry/exit
    t_patch = sess.get_patch_times()

    # Determine number of licks per patch/interpatch
    n_patch = in_interval(t_decision, 
                          t_patch[:, 0], 
                          t_patch[:, 1],
                          query='interval')
    n_interpatch = in_interval(t_decision, 
                               t_patch[:, 1], 
                               np.append(t_patch[1:, 0], sess.vars['t_stop']),
                               query='interval')
    
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

    # Determine total number of licks
    n_total = np.sum(n_patch) + np.sum(n_interpatch)

    # Save stats in dictionary
    lick_stats = {'n_total': n_total,
                  'n_patch': n_patch,
                  'n_interpatch': n_interpatch,
                  'f_total': n_total / sess.vars['t_stop'],
                  'f_patch': f_patch,
                  'f_interpatch': f_interpatch}
    
    return lick_stats

def get_entry_stats(sess, per_patch=True):

    # Get timestamps associated with lick decisions
    t_decision = get_lick_decisions(sess)

    # Get timestamps associated with patch entry/exit
    t_patch = sess.get_patch_times()

    # Find first lick after patch entry
    in_patch = in_interval(t_decision, 
                           t_patch[:, 0], 
                           t_patch[:, 1], 
                           query='array')
    idx_patch, idx_ = np.unique(np.argwhere(in_patch)[:, 0], return_index=True)
    idx_lick = np.argwhere(in_patch)[idx_, 1]

    # Calculate time to first lick from patch entry
    t_delay = t_decision[idx_lick] - t_patch[idx_patch, 0]

    if per_patch:
        return t_delay
    else:
        return np.median(t_delay)

def get_exit_stats(sess, per_patch=True):

    # Get timestamps associated with lick decisions and rewards
    t_decision = get_lick_decisions(sess)
    t_reward = sess.vars['t_motor']

    # Get timestamps associated with patch entry/exit
    t_patch = sess.get_patch_times()

    # Find last lick before patch exit
    in_patch = in_interval(t_decision, 
                           t_patch[:, 0], 
                           t_patch[:, 1], 
                           query='array')
    idx_patch, idx_, counts = np.unique(np.argwhere(in_patch)[:, 0], 
                                        return_index=True, 
                                        return_counts=True)
    idx_ += counts - 1
    idx_lick = np.argwhere(in_patch)[idx_, 1]

    # Calculate time to last lick before patch exit
    t_leave_lick = t_patch[idx_patch, 1] - t_decision[idx_lick] 

    # Find last reward before patch exit
    in_patch = in_interval(t_reward, 
                           t_patch[:, 0], 
                           t_patch[:, 1], 
                           query='array')
    idx_patch, idx_, counts = np.unique(np.argwhere(in_patch)[:, 0], 
                                        return_index=True, 
                                        return_counts=True)
    idx_ += counts - 1
    idx_reward = np.argwhere(in_patch)[idx_, 1]

    # Calculate time to last reward before patch exit
    t_leave_reward = t_patch[idx_patch, 1] - t_reward[idx_reward]

    if per_patch:
        return t_leave_lick, t_leave_reward
    else:
        return np.median(t_leave_lick), np.median(t_leave_reward)

def get_interpatch_move_times(sess, v_run=None, v_stop=0.5):
    """
    Returns timestamps associated with the onset of running
    after stopping in between patches.
    """
    # Settings
    if v_run is None:
        v_run = sess.settings['run_config']['v_leave']

    # Load data
    sess.load_data(['wheel_position'])
    sess.load_vars(['v_smooth', 't_wheel'])
    x_wheel = sess.data['wheel_position']
    v_smooth = sess.vars['v_smooth']
    t_wheel = sess.vars['t_wheel']

    # Initialize values
    in_patch = True # task starts in a patch
    stopped_in_interpatch = False
    x_start = 0.0 # linear position of current patch start
    t_move = [] # timestamp associated with initiation of movement in interpatch

    # Iterate through wheel data
    for i in range(v_smooth.shape[0]):
        # Leave patch criteria:
        # 1) in a patch
        # 2) smoothed velocity exceeds threshold
        if in_patch and v_smooth[i] > v_run:
            in_patch = False
            x_start = x_wheel[i]

        elif (stopped_in_interpatch
              and v_smooth[i] > v_run):
              stopped_in_interpatch = False
              t_move.append(t_wheel[i])
            
        # Enter patch criteria:
        # 1) not in a patch
        # 2) smoothed velocity falls below threshold
        # 3) have covered minimum interpatch distance
        elif (not in_patch 
              and v_smooth[i] < v_stop):
              if x_wheel[i] - x_start > sess.d_interpatch:
                  in_patch = True
              else:
                  stopped_in_interpatch = True

    return np.array(t_move)