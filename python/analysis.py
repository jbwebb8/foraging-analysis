from helper import _check_list

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
    for i in range(idx_max + 1, len(hist)):
        if hist[i] < (0.1 * h_max):

    
    bin_edges[np.argmax(hist)] + 0.5*bin_width



    
def get_lick_stats(sess):
    """
    Returns d` and bias for licking in vs. out of patch.
    """



    
