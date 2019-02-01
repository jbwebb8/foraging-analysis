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

def _check_list(names):
    if not isinstance(names, list):
        names = [names]
    return names

def get_lick_decisions(sess, min_interval=None):
    """
    Returns timestamps for decisions to lick during the session. These are
    defined by any lick with a preceding inter-lick interval that is greater
    than the inter-lick interval for lick-bouts.
    """
    # Get lick timestamps
    t_lick = sess.get_lick_times()
    dt_lick = np.diff(t_lick)

    # Get histogram of inter-lick intervals <= 1.0 seconds
    hist, bin_edges = np.histogram(dt_lick, range=[0.0, 1.0], bins=40)
    bin_width = bin_edges[1] - bin_edges[0] # 50 ms
    h_max = 0.0
    for i, h in enumerate(hist):


    
def get_lick_stats(sess):
    """
    Returns d` and bias for licking in vs. out of patch.
    """



    
