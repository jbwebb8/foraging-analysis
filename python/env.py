

class PatchEnvironment:
    
    def __init__(self,
                 d_interpatch, 
                 r_init, 
                 decay=0.1,
                 nc_avg=1.0,
                 target_duration=1.0,
                 max_trial_duration=4.0,
                 iti=1.5,
                 fa_timeout=4.0,
                 end_patch_type=None,
                 end_patch_val=None):
        
        # Patch settings
        self.d_interpatch = d_interpatch
        self.r_init = r_init
        self.decay = decay
        self.nc_avg = nc_avg
        self.target_duration = target_duration
        self.max_trial_duration = max_trial_duration
        self.iti = iti
        self.fa_timeout = fa_timeout
        self.end_patch_type = end_patch_type
        if isinstance(end_patch_val, list):
            self.end_patch_params = end_patch_val
            self.end_patch_val = randint(end_patch_val[0], end_patch_val[1])
        else:
            self.end_patch_params = None
            self.end_patch_val = end_patch_val
        
        # Tracking indices
        self.trial = 0
    
    @property
    def info(self):
        return {'d_interpatch': self.d_interpatch,
                'r_init': self.r_init,
                'decay': self.decay,
                'nc_avg': self.nc_avg,
                'target_duration': self.target_duration,
                'max_trial_duration': self.max_trial_duration,
                'intertrial_interval': self.iti,
                'fa_timeout': self.fa_timeout,
                'end_patch_type': self.end_patch_type,
                'end_patch_params': self.end_patch_params,
                'end_patch_val': self.end_patch_val}
    
    def reset_patch(self):
        self.trial = 0
        if self.end_patch_params is not None:
            self.end_patch_val = randint(self.end_patch_params[0], self.end_patch_params[1])
    
    def create_trial(self):
        """
        Creates trial based on patch settings. If patch is depleted, then return no targets.
        Otherwise, return delay to target sampled from exponential distribution (flat hazard rate).
        
        Returns:
        - nc_time: duration of delay to target
        - catch_trial: True if catch trial
        """
        if self.is_patch_depleted():
            return self.max_trial_duration, False
        else:
            lambda_ = 1.0 / self.nc_avg
            nc_time = -(1.0/lambda_) * math.log(1.0 - random()) # inverse transform sampling of exponential distribution
            
            if nc_time > self.max_trial_duration:
                nc_time = self.max_trial_duration
                catch_trial = True
            else:
                catch_trial = False
            
            return nc_time, catch_trial
    
    def give_reward(self):
        # Give current reward volume based on reward function and patch end function
        if self.is_patch_depleted():
            return 0.0
        else:
            return self._get_reward_volume()
    
    def increment_counter(self, n=1):
        self.trial += n
    
    def _get_reward_volume(self):
        return self.r_init * (1 - self.decay)**self.trial
    
    def is_patch_depleted(self):
        if self.end_patch_type == None:
            return False
        elif self.end_patch_type == 'reward':
            return self._get_reward_volume() < self.end_patch_val
        elif self.end_patch_type == 'trial':
            return self.trial >= self.end_patch_val
        else:
            raise ValueError('Unknown end patch type %s' % self.end_patch_type)