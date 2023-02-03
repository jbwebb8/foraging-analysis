def simulate_environment(num_steps, mouse, patch, verbose=False):

    n_patch = 0 # trial number
    t_patch = 0.0 # time in patch
    t_total = 0.0 # total time
    r_total = 0.0 # total reward harvested
    num_patches = 1 # number of patches explored
    r_last = math.inf # last reward in patch

    for i in range(num_steps):

        # Make stay or go decision
        leave = mouse.leave_patch(r_last, t_patch, n_patch)

        # If leave, incur travel cost and reset patch
        if leave:
            r_step = 0.0
            t_step = mouse.get_travel_time(patch.d_interpatch)
            n_patch = 0
            t_patch = 0.0
            r_last = math.inf
            num_patches += 1
            patch.reset_patch()

        # Otherwise, attempt one block to harvest reward
        else:
            # Get results of block
            r_step, t_step = _run_detection_block(mouse, patch)

            # Increment patch counters
            patch.increment_counter()
            n_patch += 1
            t_patch += t_step

        # Increment global counters
        t_total += t_step
        r_total += r_step

        # Print progress
        if verbose and (i % 100 == 0):
            print("Step %d of %d" % (i+1, num_steps))
    
    # Reset patch object
    patch.reset_patch()
    
    if verbose:
        harvest_rate = r_total / t_total
        print("Total reward: %.2f uL" % r_total)
        print("Total time: %.2f s" % t_total)
        print("Harvest rate: %.3f uL/s" % harvest_rate)
    
    return r_total, t_total, num_patches

def _run_detection_block(mouse, patch):
    # Get trial from patch
    nc_time, catch_trial = patch.create_trial()
    result, t_result = mouse.harvest_reward(nc_time, patch.target_duration, catch_trial)

    # Penalize with timeout if false alarm
    if (result == 2):
        r_step = 0.0
        t_step = t_result + patch.fa_timeout + patch.iti

    # Get reward if hit or successful catch trial
    elif (result == 0) or (result == 3): 
        r_step = patch.give_reward()
        r_last = r_step
        t_step = nc_time + patch.target_duration * (1 - catch_trial) + patch.iti

    # Do not reward but do not penalize if miss
    elif (result == 1):
        r_step = 0.0
        t_step = nc_time + patch.target_duration + patch.iti

    else:
        raise ValueError('Unknown result index: %d' % result)

    return r_step, t_step

def _run_free_block(mouse, patch):
    """One attempt to harvest reward in free-licking task."""
    pass