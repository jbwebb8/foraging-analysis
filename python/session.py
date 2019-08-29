# Ignore annoying FutureWarning from h5py
import warnings
warnings.simplefilter('ignore', category=FutureWarning)

# Imports
import h5py
import re
import pickle
import numpy as np
from scipy.interpolate import interp1d
from helper import smooth_waveform_variance, find_threshold, \
                   cumulative_reward, get_optimal_values, compare_patch_times
from util import in_interval, _check_list, dec_to_bin_array, flatten_list

class Session:

    def __init__(self):
        """
        Generic session class for foraging behavior. Minimum required data:
        - 
        """
        
        # Set allowed data and variable names
        self.data_names = ['motor', 'lick', 'fs', 'reward']
        self.var_names = ['T', 't_patch', 'in_patch', 't_stop', 
                          't_motor', 'dt_motor', 'n_motor_rem',
                          't_lick']

        # Set placeholders
        self.data = {}
        self.vars = {}

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
    
    @property
    def day(self):
        raise NotImplementedError

    @property
    def total_time(self):
        # Requirements
        req_vars = ['T']
        self._check_attributes(var_names=req_vars)

        return self.vars['T']

    @property
    def analyzed_time(self):
        # Requirements
        req_vars = ['t_stop']
        self._check_attributes(var_names=req_vars)

        return self.vars['t_stop']

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
        raise NotImplementedError

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
        raise NotImplementedError

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


    def get_patch_times(self, var_name='t_patch'):
        # Requirements
        req_vars = [var_name]
        self._check_attributes(var_names=req_vars)

        return self.vars[var_name]

    def _get_patch_times_from_signal(self, in_patch, fs):
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
        
        return t_patch, t_stop

    def get_patch_durations(self):
        # Requirements
        req_vars = ['t_patch']
        self._check_attributes(var_names=req_vars)

        return np.diff(self.vars['t_patch'], axis=1)[:, 0]

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
        if var_name in ['t_motor', 'dt_motor']:
            thresh = np.max(self.data['motor']) / 2.0 # half of 5V square wave
            idx_stop = int(self.vars['t_stop'] * self.data['fs'])
            motor = self.data['motor'][:idx_stop]
            is_pump, idx_pump_start, idx_pump_end = self._find_threshold_crossings(motor, thresh)
            t_motor = idx_pump_start / self.data['fs']
            dt_motor = (idx_pump_end - idx_pump_start) / self.data['fs']

            if var_name == 't_motor':
                return t_motor
            elif var_name == 'dt_motor':
                return dt_motor

        # Find number of rewards outside analyzable period
        # (for double checking reward logging)
        elif var_name == 'n_motor_rem':
            thresh = np.max(self.data['motor']) / 2.0 # half of 5V square wave
            idx_stop = int(self.vars['t_stop'] * self.data['fs'])
            motor = self.data['motor'][idx_stop:]
            is_pump, idx_pump_start, idx_pump_end = self._find_threshold_crossings(motor, thresh)
            
            return len(idx_pump_start)

        else:
            raise ValueError('Name \'%s\' not recognized.' % var_name)

    def _find_threshold_crossings(self, y, thresh):
        # Find where signal > thresh
        is_on = (y > thresh).astype(np.int32)

        # Find start and end of threshold crossings, correcting for end cases
        idx_on_start = (np.argwhere((is_on - np.roll(is_on, -1)) == -1) + 1).flatten()
        if is_on[0] and not is_on[-1]:
            idx_on_start = idx_on_start[:-1]
        idx_on_end = (np.argwhere((is_on - np.roll(is_on, -1)) == 1) + 1).flatten()
        if not is_on[0] and is_on[-1]:
            idx_on_end = idx_on_end[:-1]

        # Correct if session ended with signal on
        if idx_on_start.shape[0] == idx_on_end.shape[0] + 1:
            idx_on_end = np.append(idx_on_end, is_on.shape[0]-1)

        return is_on, idx_on_start, idx_on_end

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
        thresh = np.max(lick) / 2.0
        idx_licking = (lick > thresh).astype(np.int32)
        idx_lick = (np.argwhere((idx_licking - np.roll(idx_licking, -1)) == -1) + 1).flatten()
        t_lick = idx_lick / self.data['fs']

        return t_lick

    def get_reward_volumes(self):
        # Requirements
        req_data = ['reward']
        req_vars = ['t_motor', 'n_motor_rem']
        self._check_attributes(data_names=req_data, var_names=req_vars)

        # Get logged volumes
        r_log = self.data['reward'][self.data['reward'] > 0]

        # Compare motor trace to logged reward volume
        if len(r_log) != len(self.vars['t_motor']) + self.vars['n_motor_rem']:
            raise UserWarning('Logged rewards do not match motor trace.')
        r_log = r_log[:self.vars['t_motor'].shape[0]] # filters by t_stop 

        return r_log

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
            required_vars = ['t_patch', 't_motor', 'dt_motor']
            self._check_attributes(var_names=required_vars)
            if self.vars['t_motor'].size == 0:
                return np.zeros(self.vars['t_patch'].shape[0])
            
            # Filter logged reward volumes by those given
            r_log = self.get_reward_volumes()

            # Create linear map from motor duration to reward volume
            # (V = duration x flow_rate is not reliable)
            #m = ( (np.max(r_log) - np.min(r_log))
            #    / (np.max(self.vars['dt_motor']) - np.min(self.vars['dt_motor'])) )
            #r_motor = lambda dt: m*dt - m*np.max(self.vars['dt_motor']) + np.max(r_log)

            # Find patches in which rewards given
            pad = 0.5 # padding in seconds
            #idx_patch = in_interval(self.vars['t_motor'],
            #                        self.vars['t_patch'][:, 0]-pad,
            #                        self.vars['t_patch'][:, 1]+pad,
            #                        query='event')
            #if (idx_patch > 1).any():
            #    raise UserWarning('Motor times cannot be uniquely assigned'
            #                      ' to individual patches.')
            #idx_patch = idx_patch.astype(np.bool)
            idx_patch, _ = in_interval(self.vars['t_motor'],
                                       self.vars['t_patch'][:, 0]-pad,
                                       self.vars['t_patch'][:, 1]+pad,
                                       query='event_interval')


            # Calculate observed reward per patch
            r_patch_obs = np.zeros(self.vars['t_patch'].shape[0])
            for i in range(self.vars['t_patch'].shape[0]):
                #r_patch_obs[i] = np.sum(self.vars['dt_motor'][idx_patch == i]*self.data['flow_rate']) # flow rate
                #r_patch_obs[i] = np.sum(r_motor(self.vars['dt_motor'][idx_patch == i])) # linear map
                r_patch_obs[i] = np.sum(r_log[idx_patch == i]) # logged volumes

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
    
class TTSession(Session):

    SAMPLING_RATE = 500 # serial transmission rate of microprocessor (Hz)
    BOARD_IDS = [16, 17, 22, 23, 24, 25, 26, 27] # GPIO IDs
    REWARD_VOLUME = 2.0 # uL

    def __init__(self, data, params):
        """Session for data acquired on TreadmillTracker system"""

        Session.__init__(self)

        # Load files
        self.raw_data = data
        self.params = params

        # Add allowed data and variable names
        self.data_names += ['t_m', 't_os', 'GPIO', 'GPIO_labels']

        # Create settings
        self.settings = {}
        GPIO_labels = {}
        for k, v in self.params['GPIO'].items():
            if v in self.BOARD_IDS:
                GPIO_labels[k] = self.BOARD_IDS.index(v)
            else:
                GPIO_labels[k] = None
        self.settings['GPIO_labels'] = GPIO_labels

    @property
    def day(self):
        return self.params['Info'].get('Session', None)

    def _load_data(self, name):
        if name == 'motor':
            return self._load_motor_data()
        elif name == 'lick':
            return self._load_lick_data()
        elif name == 'fs':
            return self.SAMPLING_RATE
        elif name == 'reward':
            self._check_attributes(var_names=['t_motor', 'n_motor_rem'])
            n_motor = len(self.vars['t_motor']) + self.vars['n_motor_rem']
            return self.REWARD_VOLUME*np.ones([n_motor])
        elif name == 'GPIO':
            return dec_to_bin_array(self.raw_data[:, 1], 
                                    bits=8, 
                                    bit_order='<')
        elif name == 't_m':
            return self.raw_data[:, 0]
        elif name == 't_os':
            return self.raw_data[:, 2]

    def _load_var(self, name):
        if name == 'T':
            return self.raw_data.shape[0]
        elif name in ['t_patch', 'in_patch', 't_stop']:
            return self._get_patch_times(name)
        elif name in ['t_motor', 'dt_motor', 'n_motor_rem']:
            return self._get_motor_times(name)
        elif name == 't_lick':
            return self._get_lick_times()
        else:
            raise SyntaxError('Unknown variable name \'%s\'.' % name)
    
    def _get_GPIO_pins(self, labels):
        if not isinstance(labels, list):
            labels = [labels]

        labels = [label.lower() for label in labels]

        idx_pin = [v for k, v in self.settings['GPIO_labels'].items() 
                   if any([label.lower() in k.lower() for label in labels])]

        return np.array(idx_pin).astype(np.int64)

    def _load_motor_data(self):
        # Check requirements
        req_data = ['GPIO']
        req_vars = ['T']
        self._check_attributes(data_names=req_data, var_names=req_vars)
            
        # Get GPIO pins
        pins = self._get_GPIO_pins('dispense')

        # Add motor traces
        motor = np.zeros([self.vars['T']])
        for pin in pins:
            motor += self.data['GPIO'][:, pin]

        if (motor < 0).any() or (motor >  1).any():
            raise ValueError('Motor trace cannot be interpreted due to'
                             ' overlapping rewards and/or negative values.')
        else:
            return motor

    def _load_lick_data(self):
        # Check requirements
        req_data = ['GPIO']
        req_vars = ['T']
        self._check_attributes(data_names=req_data, var_names=req_vars)
            
        # Get GPIO pins
        pins = self._get_GPIO_pins('lick')

        # Add motor traces
        lick = np.zeros([self.vars['T']])
        for pin in pins:
            lick += self.data['GPIO'][:, pin]

        if (lick < 0).any() or (lick >  1).any():
            raise ValueError('Lick trace cannot be interpreted due to'
                             ' overlapping sensors and/or negative values.')
        else:
            return lick

    def _get_patch_times(self, var_name='t_patch'):
        # Get patch signal
        t_patch, in_patch, t_stop = self._get_patch_times_from_GPIO()

        # Return appropriate variable
        if var_name == 't_patch':
            return t_patch
        elif var_name == 'in_patch':
            return in_patch
        elif var_name == 't_stop':
            return t_stop
        else:
            raise ValueError('Name \'%s\' not recognized.' % var_name)

    def _get_patch_times_from_GPIO(self):
        # Check requirements
        req_data = ['GPIO', 'fs']
        self._check_attributes(data_names=req_data)

        # Find patch labels
        pins = self._get_GPIO_pins('poke')
        num_patches = len(pins)
        
        # Get naive patch times
        t_patch = []
        idx_patch = []
        in_patch = []
        t_stop = []
        for i, pin in enumerate(pins):
            # Get times for patch i
            in_patch_i = self.data['GPIO'][:, pin]
            t_patch_i, t_stop_i = self._get_patch_times_from_signal(in_patch_i, self.data['fs'])

            # Apply nose poke smoothing
            t_patch_i = self._smooth_patch_times(t_patch_i)

            # Log timestamps
            t_patch.append(t_patch_i)
            idx_patch.append(i*np.ones(t_patch_i.shape[0]))
            in_patch.append(in_patch_i)
            t_stop.append(t_stop_i)

        # Combine patch times and patch IDs
        t_patch = np.vstack(t_patch)
        idx_patch = np.hstack(idx_patch)
        in_patch = np.sum(np.vstack(in_patch), axis=0)
        t_stop = np.array(t_stop)

        # Sort combined data
        idx_sort = np.argsort(t_patch[:, 0])
        t_patch = t_patch[idx_sort, :]
        idx_patch = idx_patch[idx_sort]

        # Merge 

        # Keep only alternating sequences
        idx_keep = []
        for i in range(num_patches):
            # Find all occurences of patch i
            idx_i = np.argwhere(idx_patch == i).flatten()
            idx_keep_i = idx_i[np.argwhere(np.diff(idx_i) >= num_patches).flatten() + 1]
            idx_keep_i = np.insert(idx_keep_i, 0, idx_i[0])
            idx_keep.append(idx_keep_i)
        idx_keep = np.sort(flatten_list(idx_keep)).astype(np.int64)
        t_patch = t_patch[idx_keep, :]

        # Stop at t_stop
        t_stop = np.min(t_stop)
        t_patch = t_patch[t_patch[:, 1] <= t_stop, :]

        return t_patch, in_patch, t_stop

    def _smooth_patch_times(self, t_patch):
        """Applies nose poke sensor smoothing to patch entry/exit."""
        # Check shape
        if t_patch.ndim != 2:
            raise SyntaxError('t_patch must be 2D array'
                              ' but has %d dimensions.' % t_patch.ndim)
        if t_patch.shape[1] != 2:
            raise SyntaxError('t_patch must be N x 2 array' 
                              ' but is N x %d.' % t_patch.shape[1])

        # Get entry/exit delay parameters
        entry_delay = self.params['Delay']['PokeEntryDelay']/1000
        exit_delay = self.params['Delay']['PokeEntryDelay']/1000
        
        # Flatten t_patch into list
        t_patch = list(t_patch.reshape([-1], order='C'))

        # Iterate through flattened list
        i = 0 # index of current patch start
        while (i < len(t_patch)):
            if t_patch[i+1] - t_patch[i] < entry_delay:
                # Remove patch if min entry period not met
                del t_patch[i:i+2]
            elif i == len(t_patch) - 2:
                # Avoid error at end of list
                break
            elif t_patch[i+2] - t_patch[i+1] < exit_delay:
                # Merge with next patch if min exit period not met
                del t_patch[i+1:i+3]
            else:
                # Otherwise, jump to next patch
                i += 2
        
        # Return trimmed patch times as [N x 2] array
        return np.array(t_patch).reshape([-1, 2], order='C')


class LVSession(Session):

    WHEEL_CIRCUMFERENCE = 60 # cm

    def __init__(self, filename, pupil_filepath=None):
        """Session for data acquired on LabVIEW system"""

        Session.__init__(self)

        # Get filename
        self.filename = filename
        self.f = h5py.File(filename)
        self.pupil_filepath = pupil_filepath

        # Set global attributes
        match = re.search('_d[0-9]+[a-z]*_', filename, re.IGNORECASE)
        day = match.group()[2:-1]
        match = re.search('[a-z]+', day, re.IGNORECASE)
        if match is not None:
            day = day[:match.span()[0]]
        self.day = int(day)
        self.data_names += ['sound', 'cam', 'wheel_time',
                            'wheel_speed', 'wheel_position', 'dt_patch']
        self.var_names += ['s_var', 'fs_s', 't_s', 't_wheel', 'v_smooth', 
                           'd_pupil', 't_pupil']
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

    def _ASCII_to_float(self, s):
        return float([u''.join(chr(c) for c in s)][0])
    
    def _ASCII_to_string(self, s):
        return str([u''.join(chr(c) for c in s)][0])
    
    def _ASCII_to_bool(self, s):
        return bool([u''.join(chr(c) for c in s)][0])
    
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
        elif name ==  'cam':
            return self.f['UntitledCam' + chamber_number]['Data'][0, :]
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

    def _load_var(self, name):
        if name == 'T':
            return self._get_total_time()
        elif name in ['t_patch', 'in_patch', 't_stop']:
            return self._get_patch_times(name)
        elif name in ['s_var', 'fs_s', 't_s']:
            return self.preprocess_sound(name)
        elif name in ['t_motor', 'dt_motor', 'n_motor_rem']:
            return self._get_motor_times(name)
        elif name == 't_lick':
            return self._get_lick_times()
        elif name == 't_wheel':
            return self._get_wheel_times()
        elif name == 'v_smooth':
            return self._get_smoothed_velocity()
        elif name == 'd_pupil':
            return self._get_pupil_size()
        elif name == 't_pupil':
            return self._get_pupil_times()
        else:
            raise SyntaxError('Unknown variable name \'%s\'.' % name)

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

    def _get_patch_times(self, var_name='t_patch'):
        # Requirements
        required_data = ['dt_patch']
        self._check_attributes(data_names=required_data)

        t_patch, in_patch, t_stop = \
            self._get_patches_from_sound()
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

    def _get_patches_from_sound(self, auto_thresh=True, init_thresh=5.0e-9):
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
        in_patch = (s < thresh).astype(np.int32)
        t_patch, t_stop = self._get_patch_times_from_signal(in_patch, fs_s)
        dt_patch = np.diff(t_patch, axis=1).flatten()
        
        # Check for agreement with logged patch durations
        if not compare_patch_times(dt_patch, self.data['dt_patch']):
            #print('Initial threshold failed. Trying range of values...')
            
            # Try range of values for threshold until agreement reached
            thresh_range = np.linspace(0.2*thresh, 5*thresh, 50)
            for thresh in thresh_range:
                try:
                    in_patch = (s < thresh).astype(np.int32)
                    t_patch, t_stop = self._get_patch_times_from_signal(in_patch, fs_s)
                    dt_patch = np.diff(t_patch, axis=1).flatten()
                    if compare_patch_times(dt_patch, self.data['dt_patch']):
                        #print('Successful threshold found: %.2e' % thresh)
                        break
                except IndexError: # empty array handling
                    pass

        if not compare_patch_times(dt_patch, self.data['dt_patch']):
            raise UserWarning('Unable to determine patch times from sound.')
                
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

    def save(self, filepath):
        """
        Serializes dictionary of class instance to specified location.
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

    def load(self, filepath, load_keys=None, ignore_keys=[]):
        """
        Loads serialized dictionary of class instance.
        """
        f = open(filepath, 'rb')
        d = pickle.load(f)
        if load_keys is None:
            load_keys = d.keys()
        for k, v in d.items():
            if k in load_keys and not k in ignore_keys:
                self.__dict__[k] = v # avoids overwriting h5py.File
        f.close()

    def get_pupil_times(self, filepath=None):
        # Set pupil filepath if provided
        self.pupil_filepath = filepath

        # Requirements
        req_vars = ['t_pupil']
        self._check_attributes(var_names=req_vars)

        return self.vars['t_pupil']

    def get_pupil_size(self, filepath=None):
        # Set pupil filepath if provided
        self.pupil_filepath = filepath

        # Requirements
        req_vars = ['d_pupil']
        self._check_attributes(var_names=req_vars)

        return self.vars['d_pupil']

    def _get_pupil_times(self):
        # Requirements
        req_data = ['cam', 'fs']
        req_vars = ['d_pupil']
        self._check_attributes(data_names=req_data, var_names=req_vars)

        # Determine threshold crossings
        cam_sync = self.data['cam']
        _, idx_sync, _ = self._find_threshold_crossings(cam_sync, thresh=2.5)

        # Convert indices to timestamps
        t_labview = np.arange(cam_sync.shape[0]) / self.data['fs']
        t_cam = t_labview[idx_sync][-self.vars['d_pupil'].shape[0]:] # drop first x pulses

        return t_cam

    def _get_pupil_size(self):
        # Check pupil filepath
        if self.pupil_filepath is None:
            raise SyntaxError('pupil_filepath not set. Please provide keyword '
                              'argument if using get_pupil_size() or set '
                              'pupil_filepath attribute.')

        # Load raw data
        with h5py.File(self.pupil_filepath) as f:
            pupil = f['eyedata']['block0_values'][:, 0]

        # Find frames with error (size 0 or NaN)
        a = np.logical_or(pupil == 0.0, np.isnan(pupil)) # error indices
        b = ~a # no-error indices
        a = np.argwhere(a).flatten()
        b = np.argwhere(b).flatten()

        # Set error frames to nearest neighbor with correct value
        idx_nn = b[np.argmin(np.abs(a[:, np.newaxis] - b[np.newaxis, :]), axis=1)]
        pupil[a] = pupil[idx_nn]

        return pupil


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
