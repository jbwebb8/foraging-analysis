import h5py
import re
import pickle
import numpy as np
from scipy.interpolate import interp1d
from helper import smooth_waveform_variance, find_threshold, \
                   cumulative_reward, get_optimal_values, compare_patch_times
from util import in_interval, _check_list

class Session:

    WHEEL_CIRCUMFERENCE = 60 # cm

    def __init__(self, filename):
        # Get filename
        self.filename = filename
        self.f = h5py.File(filename)

        # Set global attributes
        match = re.search('_d[0-9]+[a-z]*_', filename, re.IGNORECASE)
        day = match.group()[2:-1]
        match = re.search('[a-z]+', day, re.IGNORECASE)
        if match is not None:
            day = day[:match.span()[0]]
        self.day = int(day)
        self.data_names = ['sound', 'motor', 'lick', 'fs', 'wheel_time', 
                           'wheel_speed', 'wheel_position', 'dt_patch']
        self.var_names = ['T', 't_patch', 'in_patch', 't_stop', 's_var',
                          'fs_s', 't_s', 't_motor', 'dt_motor', 't_lick',
                          't_wheel', 'v_smooth']
        self.settings = {}
        struct = self.f['Settings']['Property']

        # IDs
        self.settings['global'] = {}
        self.settings['global']['version'] = \
            self._ASCII_to_string(struct['SoftwareVersion'])
        self.settings['global']['chamber_id'] = \
            self._ASCII_to_string(struct['ChamberID'])

        # Tone cloud
        s = 'SoundConfigurationToneCloudConfig'
        self.settings['tone_cloud'] = {}
        self.settings['tone_cloud']['low_freq'] = \
            self._ASCII_to_float(struct[s + 'LowFreqHz'])
        self.settings['tone_cloud']['hi_freq'] = \
            self._ASCII_to_float(struct[s + 'HiFreqHz'])
        self.settings['tone_cloud']['num_octave'] = \
            self._ASCII_to_float(struct[s + 'FreqNumOctave'])
        self.settings['tone_cloud']['bin_width'] = \
            self._ASCII_to_float(struct[s + 'TimeBinWidthms'])
        self.settings['tone_cloud']['num_chord'] = \
            self._ASCII_to_float(struct[s + 'ToneNumChord'])

        # Target sound
        s = 'SoundConfigurationTargetSoundConfig'
        self.settings['target_sound'] = {}
        self.settings['target_sound']['low_freq'] = \
            self._ASCII_to_float(struct[s + 'TargetLowHz'])
        self.settings['target_sound']['hi_freq'] = \
            self._ASCII_to_float(struct[s + 'TargetHiHz'])
        self.settings['target_sound']['duration'] = \
            self._ASCII_to_float(struct[s + 'TargetDurationsec'])
        self.settings['target_sound']['nc_avg'] = \
            self._ASCII_to_float(struct[s + 'AvgStartsec'])
        self.settings['target_sound']['type'] = \
            self._ASCII_to_string(struct[s + 'TargetType'])

        # Set placeholders
        self.data = {}
        self.vars = {}

    def _ASCII_to_float(self, s):
        return float([u''.join(chr(c) for c in s)][0])
    
    def _ASCII_to_string(self, s):
        return str([u''.join(chr(c) for c in s)][0])
    
    def _ASCII_to_bool(self, s):
        return bool([u''.join(chr(c) for c in s)][0])

    def load_data(self, names):
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if name in self.data_names:
                if name not in self.data.keys():
                    try:
                        self.data[name] = self._load_data(name)
                    except KeyError:
                        raise UserWarning('Data \'%s\' cannot be found.' % name)
            else:
                raise Warning('Data name \'%s\' not recognized.' % name)
    
    def _load_data(self, name):
        # Get chamber number (suffix for DAQ data)
        chamber_id = self.settings['global']['chamber_id']
        if 'ephys' in chamber_id.lower():
            chamber_number = '1'
        else:
            chamber_number = self.settings['global']['chamber_id'][-1]

        # Data names
        if name == 'sound':
            return self.f['UntitledSound' + chamber_number]['Data'][0, :]
        elif name == 'lick':
            return self.f['UntitledLick' + chamber_number]['Data'][0, :]
        elif name == 'motor':
            return self.f['UntitledMotor' + chamber_number]['Data'][0, :]
        elif name == 'fs':
            return 1.0 / self.f['UntitledSound' + chamber_number]['Property']['wf_increment'][0, 0]
        elif name == 'wheel_speed':
            return self.f['UntitledWheelSpeed']['Data'][0, :] * 100 # cm/s
        elif name == 'wheel_time':
            return self.f['UntitledWheelTime']['Data'][0, :]
        elif name == 'wheel_position':
            return self.f['UntitledAngularPosition']['Data'][0, :] / 360 * self.WHEEL_CIRCUMFERENCE # cm
        elif name == 'dt_patch':
            return self.f['UntitledPatchTime']['Data'][0, :]
        else:
            return self._load_subclass_data(name)

    def _load_subclass_data(self, name):
        pass

    def clear_data(self, names=None):
        if names is not None:
            names = _check_list(names)
            for name in names:
                _ = self.data.pop(name)
        else:
            self.data = {}

    def load_vars(self, names):
        if not isinstance(names, list):
            names = [names]
        for name in names:
            if name in self.var_names:
                if name not in self.vars.keys():
                    self.vars[name] = self._load_var(name)
            else:
                raise SyntaxError('Variable name \'%s\' not recognized.' % name)

    def _load_var(self, name):
        if name == 'T':
            return self._get_total_time()
        elif name in ['t_patch', 'in_patch', 't_stop']:
            return self._get_patch_times(name)
        elif name in ['s_var', 'fs_s', 't_s']:
            return self.preprocess_sound(name)
        elif name in ['t_motor', 'dt_motor']:
            return self._get_motor_times(name)
        elif name == 't_lick':
            return self._get_lick_times()
        elif name == 't_wheel':
            return self._get_wheel_times()
        elif name == 'v_smooth':
            return self._get_smoothed_velocity()
        else:
            raise SyntaxError('Unknown variable name \'%s\'.' % name)
        
    def clear_vars(self, names=None):
        if names is not None:
            names = _check_list(names)
            for name in names:
                _ = self.vars.pop(name)
        else:
            self.vars = {}

    def _check_attributes(self, data_names=None, var_names=None):
        if data_names is not None:
            self.load_data(data_names)
        if var_names is not None:
            self.load_vars(var_names)
        if self.vars.get('t_stop', 1.0) <= 0.0:
            raise UserWarning('Unanalyzable session: not enough patches.')

    def _get_total_time(self):
        # Requirements
        req_data = ['fs']
        self._check_attributes(data_names=req_data)

        return self.f['UntitledSound' + chamber_number]['Total_Samples'][0, 0]/self.data['fs']

    def preprocess_sound(self, var_name):
        required_data = ['sound', 'fs']
        self._check_attributes(data_names=required_data)

        fs_s, t_s, s_var_smooth = smooth_waveform_variance(self.data['sound'], 
                                                           self.data['fs'])
        if var_name == 's_var':
            return s_var_smooth
        elif var_name == 'fs_s':
            return fs_s
        elif var_name == 't_s':
            return t_s
        else:
            raise ValueError('Name \'%s\' not recognized.' % var_name)

    def get_patch_times(self, var_name='t_patch'):
        # Requirements
        req_vars = [var_name]
        self._check_attributes(var_names=req_vars)

        return self.vars[var_name]

    def _get_patch_times(self, var_name='t_patch'):
        # Requirements
        required_data = ['dt_patch']
        self._check_attributes(data_names=required_data)

        t_patch, in_patch, t_stop = \
            self.get_patches_from_sound()
        dt_patch = np.diff(t_patch, axis=1).flatten()

        if compare_patch_times(dt_patch, self.data['dt_patch']):
            if var_name == 't_patch':
                return t_patch
            elif var_name == 'in_patch':
                return in_patch
            elif var_name == 't_stop':
                return t_stop
            else:
                raise ValueError('Name \'%s\' not recognized.' % var_name)
        else:
            raise UserWarning('Patch durations from sound and log file do not match.')

    def _get_patches_from_sound(self, s, fs, thresh):
        # Determine data points in patch based on sound variance
        in_patch = (s < thresh).astype(np.int32)
        
        # Find patch start points
        t_patch_start = np.argwhere((in_patch - np.roll(in_patch, -1)) == -1).flatten()
        if (in_patch[-1] == 0) and (in_patch[0] == 1):
            # Correct wrap-around error
            t_patch_start = t_patch_start[:-1]
        if (np.sum(in_patch[:2]) == 2):
            # Correct first data point
            t_patch_start = np.insert(t_patch_start, 0, 0.0)
        
        # Find patch end points
        t_patch_end = np.argwhere((in_patch - np.roll(in_patch, -1)) == 1).flatten()
        if (in_patch[-1] == 1) and (in_patch[0] == 0):
            # Correct wrap-around error
            t_patch_end = t_patch_end[:-1]
        if t_patch_end[0] < t_patch_start[0]:
            # Correct initial error
            t_patch_end = t_patch_end[1:]
        
        # Drop last patch-interpatch sequence
        if len(t_patch_start) > len(t_patch_end):
            # End in patch: drop last patch
            t_stop = t_patch_start[-1]
            t_patch_start = t_patch_start[:-1]
        elif len(t_patch_start) == len(t_patch_end):
            # End in interpatch: drop last patch and interpatch
            t_stop = t_patch_start[-1]
            t_patch_start = t_patch_start[:-1]
            t_patch_end = t_patch_end[:-1]
        else:
            raise ValueError('Timestamps not understood: %d patch starts, %d patch stops'
                            % (len(t_patch_start), len(t_patch_end)))
            
        t_patch_start = t_patch_start.astype(np.float64) / fs
        t_patch_end = t_patch_end.astype(np.float64) / fs
        t_patch = np.vstack([t_patch_start, t_patch_end]).T
        t_stop = t_stop.astype(np.float64) / fs - 0.5 # to account for error
        
        return t_patch, in_patch, t_stop

    def get_patches_from_sound(self, auto_thresh=True, init_thresh=5.0e-9):
        required_data = ['dt_patch']
        required_vars = ['s_var', 'fs_s']
        self._check_attributes(data_names=required_data,
                               var_names=required_vars)
        s = self.vars['s_var']
        fs_s = self.vars['fs_s']

        # Get initial threshold
        if auto_thresh:
            thresh = find_threshold(s)
        else:
            thresh = init_thresh
        #print('Variance threshold: %.2e' % thresh)
        
        # Attempt to extract patches with initial threshold
        t_patch, in_patch, t_stop = \
            self._get_patches_from_sound(s, fs_s, thresh=thresh)
        dt_patch = np.diff(t_patch, axis=1).flatten()
        
        # Check for agreement with logged patch durations
        if not compare_patch_times(dt_patch, self.data['dt_patch']):
            #print('Initial threshold failed. Trying range of values...')
            
            # Try range of values for threshold until agreement reached
            thresh_range = np.linspace(0.2*thresh, 5*thresh, 50)
            for thresh in thresh_range:
                try:
                    t_patch, in_patch, t_stop = \
                        self._get_patches_from_sound(s, fs_s, thresh=thresh)
                    dt_patch = np.diff(t_patch, axis=1).flatten()
                    if compare_patch_times(dt_patch, self.data['dt_patch']):
                        #print('Successful threshold found: %.2e' % thresh)
                        break
                except IndexError: # empty array handling
                    pass
                
        return t_patch, in_patch, t_stop

    def _get_patches_from_wheel(self, stop_thresh=0.5, return_idx=False):
        # Requirements
        req_data = ['wheel_position', 'wheel_time']
        req_vars = ['v_smooth']
        self._check_attributes(data_names=req_data, var_names=req_vars)

        # Load data
        v_smooth = self.vars['v_smooth']
        x_wheel = self.data['wheel_position']
        t_wheel = self.data['wheel_time']

        # Session parameters
        v_run = self.settings['run_config']['v_leave']
        v_stop = stop_thresh # threshold for stopping (cm/s)

        # Initialize values
        in_patch = True # task starts in a patch
        x_start = 0.0 # linear position of current patch start
        t_patch_wheel = [0.0] # patch entry/exit timestamps according to wheel data
        idx_patch_wheel = [0] # indices of above timestamps
        idx_in_patch = np.zeros(v_smooth.shape[0], dtype=np.bool)

        # Iterate through wheel data
        for i in range(v_smooth.shape[0]):
            # Leave patch criteria:
            # 1) in a patch
            # 2) smoothed velocity exceeds threshold
            if in_patch and v_smooth[i] > v_run:
                in_patch = False
                x_start = x_wheel[i]
                t_patch_wheel.append(t_wheel[i])
                idx_patch_wheel.append(i)
            # Enter patch criteria:
            # 1) not in a patch
            # 2) smoothed velocity falls below threshold
            # 3) have covered minimum interpatch distance
            elif (not in_patch
                and v_smooth[i] < v_stop
                and x_wheel[i] - x_start > self.d_interpatch):
                in_patch = True
                t_patch_wheel.append(t_wheel[i])
                idx_patch_wheel.append(i)
            
            # Update in_patch indices
            idx_in_patch[i] = in_patch

        # Drop last patch-interpatch sequence
        t_patch_wheel = np.array(t_patch_wheel)
        idx_patch_wheel = np.array(idx_patch_wheel)
        if t_patch_wheel.shape[0] % 2 == 1: 
            # End in patch: drop last patch entry
            t_stop = t_patch_wheel[-1]
            idx_stop = idx_patch_wheel[-1]
            t_patch_wheel = t_patch_wheel[:-1]
            idx_patch_wheel = idx_patch_wheel[:-1]
        else: 
            # End in interpatch: drop last patch entry and exit
            t_stop = t_patch_wheel[-2]
            idx_stop = idx_patch_wheel[-2]
            t_patch_wheel = t_patch_wheel[:-2]
            idx_patch_wheel = idx_patch_wheel[:-2]

        # Reshape
        t_patch_wheel = t_patch_wheel.reshape([-1, 2])
        idx_patch_wheel = idx_patch_wheel.reshape([-1, 2])

        if return_idx:
            return t_patch_wheel, idx_patch_wheel, idx_in_patch, \
                t_stop, idx_stop
        else:
            return t_patch_wheel, idx_in_patch, t_stop
        

    def get_patch_durations(self):
        # Requirements
        req_vars = ['t_patch']
        self._check_attributes(var_names=req_vars)

        return np.squeeze(np.diff(self.vars['t_patch'], axis=1))

    def get_interpatch_durations(self):
        # Requirements
        req_vars = ['t_patch', 't_stop']
        self._check_attributes(var_names=req_vars)

        t_switch = np.reshape(self.vars['t_patch'], [-1], order='C')
        t_interpatch = np.append(t_switch[1:], self.vars['t_stop'])
        t_interpatch = np.reshape(t_interpatch, [-1, 2])

        return np.squeeze(np.diff(t_interpatch, axis=1))

    def get_motor_times(self, var_name='t_motor'):
        # Requirements
        req_vars = [var_name]
        self._check_attributes(var_names=req_vars)

        return self.vars[var_name]
    
    def _get_motor_times(self, var_name='t_motor'):
        # Requirements
        req_data = ['motor', 'fs']
        req_vars = ['t_stop']
        self._check_attributes(data_names=req_data, var_names=req_vars)

        # Find motor timestamps and durations
        idx_stop = int(self.vars['t_stop'] * self.data['fs'])
        motor = self.data['motor'][:idx_stop]
        thresh = 2.5 # half of 5V square wave
        is_pump = (motor > thresh).astype(np.int32)
        idx_pump_start = (np.argwhere((is_pump - np.roll(is_pump, -1)) == -1) + 1).flatten()
        idx_pump_end = (np.argwhere((is_pump - np.roll(is_pump, -1)) == 1) + 1).flatten() 
        t_motor = idx_pump_start / self.data['fs']
        dt_motor = (idx_pump_end - idx_pump_start) / self.data['fs']

        if var_name == 't_motor':
            return t_motor
        elif var_name == 'dt_motor':
            return dt_motor
        else:
            raise ValueError('Name \'%s\' not recognized.' % var_name)

    def get_lick_times(self):
        # Requirements
        req_vars = ['t_lick']
        self._check_attributes(var_names=req_vars)

        return self.vars['t_lick']
    
    def _get_lick_times(self):
        # Requirements
        req_data = ['lick', 'fs']
        req_vars = ['t_stop']
        self._check_attributes(data_names=req_data, var_names=req_vars)

        # Find lick timestamps
        idx_stop = int(self.vars['t_stop'] * self.data['fs'])
        lick = self.data['lick'][:idx_stop]
        thresh = np.max(lick) / 2
        idx_licking = (lick > thresh).astype(np.int32)
        idx_lick = (np.argwhere((idx_licking - np.roll(idx_licking, -1)) == -1) + 1).flatten()
        t_lick = idx_lick / self.data['fs']

        return t_lick

    def get_wheel_times(self):
        # Requirements
        req_vars = ['t_wheel']
        self._check_attributes(var_names=req_vars)

        return self.vars['t_wheel']

    def _get_wheel_times(self):
        # Requirements
        req_data = ['wheel_time']
        req_vars = ['t_patch', 't_stop']
        self._check_attributes(data_names=req_data, var_names=req_vars)

        # Get patch entry/exit timestamps from sound
        t_patch_sound = self.vars['t_patch']

        # Get patch entry/exit timestamps from wheel
        t_patch_wheel, idx_patch_wheel, _, t_stop, idx_stop = \
            self._get_patches_from_wheel(return_idx=True)
        t_wheel = self.data['wheel_time']

        # Load last analyzable timestamp for interpolation function
        # (corresponds to last patch entry)
        t_stop = self.vars['t_stop']

        # Estimate offset between encoder and DAQ start based on
        # first patch exit (first analyzable timestamp)
        offset = t_patch_sound[0, 1] - t_patch_wheel[0, 1]

        # Interpolate between patch entry/exit timestamps to create
        # aligned wheel timestamps. Anchor wheel indices associated
        # with patch entry/exit (based on smoothed velocity) to 
        # corresponding timestamps based on sound. Handle special case
        # of session start (first patch "entry") due to initial offset
        # between DAQ and encoder.
        x_interp = np.append(idx_patch_wheel.flatten(), idx_stop)
        y_interp = np.append(np.insert(t_patch_sound.flatten()[1:], 0, offset), t_stop)
        f = interp1d(x_interp, y_interp)
        t_wheel_ = np.zeros(t_wheel.shape)
        t_wheel_[:idx_stop+1] = f(np.arange(idx_stop+1))

        # Assign times after last patch entry (t_stop) based
        # on original wheel timestamps
        t_wheel_[idx_stop+1:] = t_wheel_[idx_stop] + (t_wheel[idx_stop+1:] - t_wheel_[idx_stop])

        return t_wheel_

    def _get_smoothed_velocity(self, dt=1.0, fs=200.0):
        # Requirements
        req_data = ['wheel_speed']
        self._check_attributes(data_names=req_data)

        # Create smoothed velocity trace
        num_smooth = dt * fs # number of previous samples to include for smoothing
        v_wheel = self.data['wheel_speed']
        v_smooth = np.zeros(v_wheel.shape)
        for i in range(v_wheel.shape[0]):
            i_last = int(max(0, i - num_smooth + 1))
            v_smooth[i] = np.mean(v_wheel[i_last:i+1])
        
        return v_smooth

    def get_harvest_rate(self, metric='observed', **kwargs):
        if metric == 'observed':
            return self._get_observed_harvest_rate(**kwargs)
        elif metric == 'max':
            return self._get_max_harvest_rate(**kwargs)
        elif metric == 'optimal':
            return self._get_optimal_harvest_rate(**kwargs)
    
    def _get_observed_harvest_rate(self, per_patch=True, return_all=False):
        if per_patch:
            # Requirements
            required_data = ['reward']
            required_vars = ['t_patch', 't_motor', 'dt_motor']
            self._check_attributes(data_names=required_data, var_names=required_vars)
            if self.vars['t_motor'].size == 0:
                return np.zeros(self.vars['t_patch'].shape[0])
            
            # Create linear map from motor duration to reward volume
            # (V = duration x flow_rate is not reliable)
            r_log = self.data['reward'][self.data['reward'] > 0]
            m = ( (np.max(r_log) - np.min(r_log))
                / (np.max(self.vars['dt_motor']) - np.min(self.vars['dt_motor'])) )
            r_motor = lambda dt: m*dt - m*np.max(self.vars['dt_motor']) + np.max(r_log)

            # Just compare motor trace to logged reward volume
            #r_log = self.data['reward'][self.data['reward'] > 0]
            # TODO: correct for rewards logged after t_stop
            #r_log = r_log[:self.vars['t_motor'].shape[0]]

            # Find patches in which rewards given
            pad = 0.5 # padding in seconds
            #gt_t1 = self.vars['t_motor'][np.newaxis, :] > self.vars['t_patch'][:, 0, np.newaxis] - pad
            #lt_t2 = self.vars['t_motor'][np.newaxis, :] < self.vars['t_patch'][:, 1, np.newaxis] + pad
            #idx_patch = np.argwhere(np.logical_and(gt_t1, lt_t2))[:, 0]
            idx_patch = in_interval(self.vars['t_motor'],
                                    self.vars['t_patch'][:, 0],
                                    self.vars['t_patch'][:, 1],
                                    query='event')
            if (idx_patch > 1).any():
                raise UserWarning('Motor times cannot be uniquely assigned'
                                  ' to individual patches.')
            idx_patch = idx_patch.astype(np.bool)

            # Calculate observed reward per patch
            r_patch_obs = np.zeros(self.vars['t_patch'].shape[0])
            for i in range(self.vars['t_patch'].shape[0]):
                #r_patch_obs[i] = np.sum(self.vars['dt_motor'][idx_patch == i] * self.data['flow_rate'])
                r_patch_obs[i] = np.sum(r_motor(self.vars['dt_motor'][idx_patch == i]))
                #r_patch_obs[i] = np.sum(r_log[idx_patch == i])

            # Divide reward per patch by segment time (patch + next interpatch)
            t_seg = self._get_segment_durations()
            hr_patch_obs = r_patch_obs / t_seg

            if return_all:
                return hr_patch_obs, r_patch_obs, t_seg
            else:
                return hr_patch_obs
        
        else:
            # Requirements
            required_data = ['reward']
            required_vars = ['t_stop']
            self._check_attributes(data_names=required_data, var_names=required_vars)

            # Get total time and reward
            t_total = self.vars['t_stop']
            r_total = np.sum(self.data['reward'])
            
            if return_all:
                return r_total / t_total, r_total, t_total
            else:
                return r_total / t_total

    def _get_max_harvest_rate(self, per_patch=True):
        """
        Returns maximum attainable harvest rate given limitations of the
        animal for the given task.
        """
        raise NotImplementedError

    def _get_optimal_harvest_rate(self, per_patch=True):
        """
        Returns maximum theoretical harvest rate according to MVT.
        """
        raise NotImplementedError

    def _get_segment_durations(self):
        # Requirements
        required_vars = ['t_patch', 't_stop']
        self._check_attributes(var_names=required_vars)

        t_seg = np.diff(self.vars['t_patch'][:, 0])
        t_seg_last = self.vars['t_stop'] - self.vars['t_patch'][-1, 0]

        return np.append(t_seg, t_seg_last)

    @property
    def n_patches(self):
        """
        Returns number of analyzable patch-interpatch segments
        during session.
        """
        # Requirements
        required_vars = ['t_patch']
        self._check_attributes(var_names=required_vars)

        return self.vars['t_patch'].shape[0]

    def save(self, filepath):
        """
        Serializes dictionary of class instance to specificed location.
        Can be subsequently loaded via:
            sess.save(filepath)
            new_sess = Session(data_file)
            new_sess.load(filepath)
        """
        f = open(filepath, 'wb')
        d = self.__dict__.copy()
        _ = d.pop('f', None) # cannot pickle h5py
        pickle.dump(d, f)
        f.close()

    def load(self, filepath):
        """
        Loads serialized dictionary of class instance.
        """
        f = open(filepath, 'rb')
        d = pickle.load(f)
        for k, v in d.items():
            self.__dict__[k] = v # avoids overwriting h5py.File
        f.close()


class FreeSession(Session):

    def __init__(self, filename):
        # Initialize Session
        super().__init__(filename)

        # Set class instance attributes
        self.data_names += ['reward']
        struct = self.f['Settings']['Property']

        # Run configuration
        s = 'SoundConfigurationRunConfig'
        self.settings['run_config'] = {}
        self.settings['run_config']['session_duration'] = \
            self._ASCII_to_float(struct[s + 'SessionTimemin'])
        self.settings['run_config']['noise_level'] = \
            self._ASCII_to_float(struct[s + 'NoiseLeveldBSPL'])
        self.settings['run_config']['d_interpatch'] = \
            self._ASCII_to_float(struct[s + 'InterPatchDistcm'])
        self.settings['run_config']['d_patch'] = \
            self._ASCII_to_float(struct[s + 'PatchLengthcm'])
        self.settings['run_config']['task_type'] = \
            self._ASCII_to_string(struct[s + 'TaskType'])
        self.settings['run_config']['teleport'] = \
            self._ASCII_to_bool(struct[s + 'InPatchTeleport'])
        self.settings['run_config']['teleport_length'] = \
            self._ASCII_to_float(struct[s + 'TeleToPatchEndcm'])
        self.settings['run_config']['r_init'] = \
            self._ASCII_to_float(struct[s + 'IniVoluL'])
        self.settings['run_config']['rate_init'] = \
            self._ASCII_to_float(struct[s + 'IniRateuLsec'])
        self.settings['run_config']['tau'] = \
            self._ASCII_to_float(struct[s + 'TCsec'])
        #self.settings['run_config']['r_low'] = \
        #    self._ASCII_to_float(struct[s + 'ThresholduL'])
        self.settings['run_config']['end_target_trial'] = \
            self._ASCII_to_bool(struct[s + 'Endtargettrial'])
        self.settings['run_config']['v_leave'] = \
            self._ASCII_to_float(struct[s + 'VThresholdms']) * 100 # cm/s
        self.settings['run_config']['end_patch_speed'] = \
            self._ASCII_to_bool(struct[s + 'Endpatchspeed'])
    
    def _load_subclass_data(self, name):
        if name == 'reward':
            return np.squeeze(self.f['UntitledRewarduL']['Data'])
    
    def _get_max_harvest_rate(self, per_patch=True, return_all=False):
        # Requirements
        required_vars = ['t_patch', 't_stop']
        self._check_attributes(var_names=required_vars)

        # Grab session settings
        R_0 = self.settings['run_config']['r_init']
        r_0 = self.settings['run_config']['rate_init']
        tau = self.settings['run_config']['tau']

        # Calculate cumulative reward
        dt_patch = np.diff(self.vars['t_patch'], axis=1)
        r_patch_max = cumulative_reward(dt_patch, R_0, r_0, tau)

        if per_patch:
            # Divide reward per patch by segment time (patch + next interpatch)
            hr_patch_max = r_patch_max / self._get_segment_durations()
            if return_all:
                return hr_patch_max, r_patch_max, self._get_segment_durations()
            else:
                return hr_patch_max

        else:
            # Return total cumulative reward over time
            hr_max = np.sum(r_patch_max) / self.vars['t_stop']
            if return_all:
                return hr_max, np.sum(r_patch_max), self.vars['t_stop']
            else:
                return hr_max

    def _get_optimal_harvest_rate(self, per_patch=True, return_all=False):
        # Requirements
        required_data = ['wheel_speed']
        self._check_attributes(data_names=required_data)

        # Calculate minimum feasible travel time
        v_thresh = 2.0 # cm/s 
        d_interpatch = self.settings['run_config']['d_interpatch']
        v_run = self.data['wheel_speed']
        v_run = np.median(v_run[v_run > v_thresh])
        t_t = d_interpatch / v_run

        # Grab session settings
        R_0 = self.settings['run_config']['r_init']
        r_0 = self.settings['run_config']['rate_init']
        tau = self.settings['run_config']['tau']

        # Minimum travel time: R_0 / r_0
        t_p_opt, r_opt = get_optimal_values(t_t=t_t, R_0=R_0, r_0=r_0, tau=tau)
        
        if return_all:
            return r_opt / (t_p_opt + t_t), r_opt, t_p_opt, t_t
        else:
            return r_opt / (t_p_opt + t_t)
    
    @property
    def d_interpatch(self):
        return self.settings['run_config']['d_interpatch']
    
    @property
    def d_patch(self):
        return self.settings['run_config']['d_patch']


class TrialSession(Session):

    def __init__(self, filename):
        # Initialize Session
        super.__init__(self, filename)

        # Set class instance attributes
        with h5py.File(filename) as f:
            struct = f['Settings']['Property']

            # Run configuration
            s = 'SoundConfigurationRunConfig'
            self.settings['run_config']['max_trial_duration'] = \
                self._ASCII_to_float(struct[s + 'TrialDurationMaxsec'])
            self.settings['run_config']['catch_trial'] = \
                self._ASCII_to_bool(struct[s + 'CatchTrial'])
            self.settings['run_config']['session_duration'] = \
                self._ASCII_to_float(struct[s + 'SessionTimemin'])
            self.settings['run_config']['noise_level'] = \
                self._ASCII_to_float(struct[s + 'NoiseLeveldBSPL'])
            self.settings['run_config']['fa_timeout'] = \
                self._ASCII_to_float(struct[s + 'FalseAlarmTimeoutsec'])
            self.settings['run_config']['target_level'] = \
                self._ASCII_to_float(struct[s + 'TargetLevelHidBorDP'])
            self.settings['run_config']['target_step'] = \
                self._ASCII_to_float(struct[s + 'StepdBorDP'])
            self.settings['run_config']['d_interpatch'] = \
                self._ASCII_to_float(struct[s + 'InterPatchDistcm'])
            self.settings['run_config']['d_patch'] = \
                self._ASCII_to_float(struct[s + 'PatchLengthcm'])
            self.settings['run_config']['task_type'] = \
                self._ASCII_to_string(struct[s + 'TaskType'])
            self.settings['run_config']['teleport'] = \
                self._ASCII_to_bool(struct[s + 'InPatchTeleport'])
            self.settings['run_config']['teleport_length'] = \
                self._ASCII_to_float(struct[s + 'TeleToPatchEndcm'])
            self.settings['run_config']['n_trials_low'] = \
                self._ASCII_to_float(struct[s + 'TrialLow'])
            self.settings['run_config']['n_trials_high'] = \
                self._ASCII_to_float(struct[s + 'TrialHigh'])
            self.settings['run_config']['r_init'] = \
                self._ASCII_to_float(struct[s + 'RewarduL'])
            self.settings['run_config']['decay'] = \
                self._ASCII_to_float(struct[s + 'Decay'])
            self.settings['run_config']['r_low'] = \
                self._ASCII_to_float(struct[s + 'ThresholduL'])
            self.settings['run_config']['end_target_trial'] = \
                self._ASCII_to_bool(struct[s + 'Endtargettrial'])
            self.settings['run_config']['end_target_reward'] = \
                self._ASCII_to_bool(struct[s + 'Endtargetreward'])
            self.settings['run_config']['v_leave'] = \
                self._ASCII_to_float(struct[s + 'VThresholdms'])
            self.settings['run_config']['end_patch_speed'] = \
                self._ASCII_to_bool(struct[s + 'Endpatchspeed'])
