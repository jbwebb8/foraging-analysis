import copy
import itertools, functools
import numpy as np
import scipy.optimize as opt
from scipy import integrate
from scipy.special import factorial
import ephys
import util
import helper

# ForagingModel
# --HeuristicModel
#   --RewardTime
#   --RewardNumber
#   --TimeOnTask
# --MVTModel
#   --VanillaMVT
#   --FittedMVT
# --BayesianModel
#   --VanillaBayes
#   --FittedBayes
# --RLModel

# ForagingModel
#     __init__(): initialize model and variables
#     fit(): fit model to data, where data includes patch residence times
#         and/or reward sequences. heuristic models: determine thresholds;
#         MVT models: determine Tp and, if FittedMVT, fit parameter(s); Bayesian
#         models: set estimated parameters and lambda threshold
#     predict(): predict residence times from environmental conditions and/or
#         reward sequences (useful for cross-validation or computing MSE)

# NOTES
# - This structure should easily allow time-on-task effects to be accounted for by
#   giving each model its own TimeOnTask model to subtract off effect.
# - Let's opt for a one-model-per-subject approach (as well as a model for pooled data),
#   instead of creating one model to handle all subjects (basically handling dictionaries
#   of data).
# - Filter data by included conditions prior to creating/fitting model.
# - Environment parameters should be of the form:
#   {num: {'tau': value, 'V0': value, 'lam0': value, 'track': value}, ...}
# - New idea: instead of setting all parameters when switching environments, simply
#   create a submodel for each environment, each of which has its unique set of 
#   parameters and data. Then, switching environments simply amounts to switching
#   which model is "active". To implement this, we may need to restructure classes
#   to have some sort of manager which oversees all the submodels. Maybe all foraging
#   models are built from a Model class that takes a type argument in the initialization
#   to build a list of specific submodel classes.
# - I've gone back and forth between two different formats of the data subsets:
#   
#   1) data[<(mouse_id, env_id)>][<data_key>] = list of animal/condition session data
#   2) data[<data_key>] = list of all session data
#   
#   I think the second format is probably better in that it allows the predict() method
#   to more easily be extrapolated to using data exterior to the model. That is, one can
#   simply feed in a list of session data rather than relying on the internal representation
#   of the test dataset, whereas with the first format, the model would expect a dictionary
#   object with data organized by (mouse_id, env_id) tuple keys.

class ForagingModel:
    
    REWARD_VOLUME = 2.0 # volume of rewards (uL)
    ENV_KEYS = ['tau', 'V0', 'lambda0', 'track']
    RANDOM_SEED = 12345
    MAX_WAIT_TIME = 10.0 # maximum waiting time for reward if not modeled

    def __init__(self):
        # Placeholders for data and model parameters
        # Each item will be specific to a combination of animal ID and env ID:
        # data[(mouse_id, cond_id)] = vals
        # where vals is a list of values in which each item represent data for
        # a given session for animal mouse_id under condition cond_id.
        self._data = {}
        self._data_keys = []
        self._params = {}
        self._param_keys = []

        # Set environment parameters
        self.env_params = {} # dict of {env_number: params}
        #self.set_environment(list(env_params.keys())[0]) # current environment
        self._env = None
        self._patch_model = None

        # Placeholder for all and current animal(s) and condition(s) to model
        self._mice = []
        self._envs = []
        self._active_mice = []
        self._active_envs = []

        # Initialize data subsets
        self._k = -1 # forces subset creation with first call to add_data()

        # Placeholder for cached computationally expensive functions
        self._E_reward_time = {}

    @property
    def _active_key(self):
        return tuple(self._active_mice + self._active_envs 
                     + ['k', self._k, 'index', self._test_index])

    @property
    def params(self):
        return self._params[self._active_key]

    @property
    def data(self):
        return self._data

    @property
    def _active_data(self):
        data = {k: [] for k in self._data_keys}
        for k in self._data_keys:
            for key in itertools.product(self._active_mice, self._active_envs):
                if key in self._data.keys():
                    data[k] += self._data[key][k]
        return data

    @property
    def train_data(self):
        return self._get_subset_data(dataset='train')

    @property
    def test_data(self):
        return self._get_subset_data(dataset='test')

    def _get_subset_data(self, dataset):
        # Get relevant indices
        if dataset == 'train':
            compare = lambda i, j: i != j
        elif dataset == 'test':
            compare = lambda i, j: i == j
        else:
            raise ValueError('Dataset {} not recognized.'.format(dataset))

        # Check for proper initialization of data keys
        if len(self._data_keys) == 0:
            raise KeyError('Data keys not initialized.')
        
        # Format 1: data[<data_key>] = list of pooled session data
        # Add data if 1) involves active mice and environment, and 2) is part of the
        # queried data subset (i.e. train or test).
        N = None
        data = {k: [] for k in self._data_keys + ['mouse_id', 'env_id']}
        for key in itertools.product(self._active_mice, self._active_envs):
            if key in self._data.keys():
                subset = []
                # Iterate over sessions
                for i, subset_idx in enumerate(self._subset_idx[key]):
                    # Iterate over within-session split
                    for j, idx in enumerate(subset_idx):
                        if compare(j, self._test_index):
                            for k in self._data_keys:
                                # data[(mouse_id, env_id)][<data_name>][<session>][<subset>]
                                if isinstance(self._data[key][k][i], np.ndarray):
                                    data[k].append(self._data[key][k][i][idx])
                                else:
                                    data[k].append([self._data[key][k][i][n] for n in idx])
                    
                            # Add animal and environment IDs associated with session
                            data['mouse_id'].append(key[0])
                            data['env_id'].append(key[1])

        
        # Add associated animal and environment IDs
        # for key in itertools.product(self._active_mice, self._active_envs):
        #     if key in self._data.keys():
        #         data['mouse_id'] += [key[0]]*N
        #         data['env_id']   += [key[1]]*N
        
        # Format 2: data[<mouse_id, env_id>][<data_key>] = list of session data
        # data = {}
        # for key in itertools.product(self._active_mice, self._active_envs):
        #     if key in self._data.keys():
        #         data[key] = {k: [] for k in self._data_keys}
        #         for k in self._data_keys:
        #             subset = []
        #             # Iterate over sessions
        #             for i, subset_idx in enumerate(self._subset_idx[key]):
        #                 # Iterate over within-session split
        #                 for j, idx in enumerate(subset_idx):
        #                     if compare(j, self._test_index):
        #                         # data[(mouse_id, env_id)][<data_name>][<session>][<subset>]
        #                         if isinstance(self._data[key][k][i], np.ndarray):
        #                             subset.append(self._data[key][k][i][idx])
        #                         else:
        #                             subset.append([self._data[key][k][i][n] for n in idx])
        #             data[key][k] += subset

        return data

    def unravel(self, data, *args):
        assert all([len(data) == len(arg) for arg in args])
        if all([isinstance(dat, (list, np.ndarray)) for dat in data]):
            new_data = []
            new_args = [[] for i in range(len(args))]
            for i, dat in enumerate(data):
                new_data += [d for d in dat] # works for list or array
                for j, arg in enumerate(args):
                    if isinstance(arg[i], (list, np.ndarray)):
                        assert len(arg[i]) == len(dat)
                        new_args[j] += [d for d in arg[i]]
                    else:
                        new_args[j] += [arg[i]]*len(dat)
            if len(args) > 0:
                return new_data, tuple(new_args)
            else:
                return new_data
        else:
            if len(args) > 0:
                return data, args
            else:
                return data

    def add_environment(self, env_id, env_params):
        if env_id not in self._envs:
            # Check formatting
            env_params = self._check_env_params(env_params)

            # Add environment to model
            self._envs.append(env_id)
            self.env_params[env_id] = env_params
            if self._env is None:
                self.set_environment(env_id)
    
    def _check_env_params(self, env_params):
        # Check for required keys
        missing_keys = [key for key in self.ENV_KEYS if key not in env_params]
        if len(missing_keys) > 0:
            raise ValueError('Environment parameters must include {}.'
                             .format(', '.join(missing_keys)))
        
        # Freely-moving data conversion for convenience
        if env_params['track'] == 'LinearTrack-1m':
            env_params['track'] = 100.0
        elif env_params['track'] == 'LTrack-4m':
            env_params['track'] = 400.0

        return env_params

    def set_environment(self, env):
        # Check environment ID
        if env not in self._envs:
            raise ValueError('Environment {} not yet added.'.format(env))

        # Set environment parameters
        self._env = env
        self._tau = self.env_params[env]['tau']
        self._V0 = self.env_params[env]['V0']
        self._lambda0 = self.env_params[env]['lambda0']
        self._track = self.env_params[env]['track']

        # Update patch model for simulations
        self._update_patch_model()

    def _update_patch_model(self):
        lam = lambda t: self._lambda0*np.exp(-t/self._tau)
        Lam = lambda t, s: self._lambda0*self._tau*(np.exp(-t/self._tau) - np.exp(-(t+s)/self._tau))
        self._patch_model = ephys.Poisson(lam=lam, Lam=Lam, homogeneous=False)

    def add_data(self, mouse_id, env_id, **kwargs):
        # Check environment
        if env_id not in self._envs:
            raise ValueError('Environment {} not yet added to model.'.format(env_id))
        
        # Add animal to model
        if mouse_id not in self._mice:
            self._mice.append(mouse_id)

        # Add data
        if (mouse_id, env_id) not in self._data.keys():
            self._data[(mouse_id, env_id)] = {}
        for key in self._data_keys:
            val = kwargs.pop(key, None)
            if val is not None:
                if key not in self._data[(mouse_id, env_id)].keys():
                    self._data[(mouse_id, env_id)][key] = [val]
                else:
                    self._data[(mouse_id, env_id)][key].append(val)

        # Reset data subset indices.
        self._reset_subsets()

    def _format_data(self, data):
        return data

    def set_data(self, mouse_ids=None, env_ids=None, k=1, index=None):
        # Set current subset of animals
        if mouse_ids is not None:
            if not isinstance(mouse_ids, list):
                mouse_ids = [mouse_ids]

            # User specifies subset of animals
            missing_ids = [m for m in mouse_ids if m not in self._mice]
            if len(missing_ids) > 0:
                raise ValueError('Animal(s) {} not yet added to model.'
                                 .format(', '.join(missing_ids)))
            self._active_mice = list(mouse_ids)
        else:
            # Otherwise, default to all animals.
            self._active_mice = copy.copy(self._mice)

        # Set current subset of environments
        if env_ids is not None:
            if not isinstance(env_ids, list):
                env_ids = [env_ids]

            # User specifies subset of environments
            missing_ids = [m for m in env_ids if m not in self._envs]
            if len(missing_ids) > 0:
                raise ValueError('Environment(s) {} not yet added to model.'
                                 .format(', '.join(missing_ids)))
            self._active_envs = list(env_ids)
        else:
            # Otherwise, default to all environments.
            self._active_envs = copy.copy(self._envs)
        
        # Split train and test datasets. Use reproducible indices to ensure
        # full coverage of dataset for cross-validation.
        if (index is not None) and (index < 0 or index > k-1):
            raise ValueError('Cross-validation index must be between 0 and k-1.')
        else:
            self._assign_subsets(k, index)

    def _assign_subsets(self, k, index):
        # Keep only indices to avoid doubly loading data into memory.
        # I think indexing the full dataset should be relatively fast,
        # so this approach wins in the memory-speed tradeoff. We need only
        # remember the index for test data.
        self._test_index = index
        
        # Create new indices if needed.
        self._split_data(k)

    def _reset_subsets(self):
        self._subset_idx = {}

    def _split_data(self, k):
        if k != self._k:
            # Reset subset indices
            self._reset_subsets()
            self._k = k

        # Set random number generator with seed for reproducibility.
        rng = np.random.default_rng(self.RANDOM_SEED)
        
        # Iterate over data[(mouse_id, env_id)][key] = [val for val in sessions]
        for (mouse_id, env_id), dat in self._data.items():
            for key, vals in dat.items():
                # Vals is list of components, where each list item represents
                # one session for that (mouse_id, env_id) combination.
                subset_idx = []
                for i, val in enumerate(vals):
                    if (mouse_id, env_id) not in self._subset_idx.keys():
                        # Randomly index session components.
                        rand_idx = rng.permutation(len(val))
                        idx = []
                        for j in range(k):
                            # Order of operations is important. If we instead grouped 
                            # the indices as (i*len(a))//k, then we wouldn't need to 
                            # add the remainder, but every len(a)//k subsets would have
                            # more data, leading to an imbalance across the data subsets.
                            # This way, we randomly add the remaining data to balance the
                            # subsets as a whole.
                            idx.append(rand_idx[j*(len(val)//k):(j+1)*(len(val)//k)])

                        # Add remaining indices randomly to subsets. 
                        for j, idx_rem in enumerate(rng.integers(len(idx), size=len(val)%k)):
                            idx[idx_rem] = np.append(idx[idx_rem], rand_idx[-j])
                        
                        # Save random permutation of indices
                        subset_idx.append(idx)
                    else:
                        # Check for equal length sequences between data types
                        size = functools.reduce(lambda count, l: count + len(l),
                                                self._subset_idx[(mouse_id, env_id)][i],
                                                0)
                        assert len(val) == size
            
                # Save subset indices for animal/condition
                if (mouse_id, env_id) not in self._subset_idx.keys():
                    self._subset_idx[(mouse_id, env_id)] = subset_idx

    def _format_input(self, tp=None, tr=None):
        pass

    def fit(self, overwrite=False, **kwargs):
        if len(self._active_envs) == 0:
            raise RuntimeError('Model environment(s) must be specified before fitting.')
        elif len(self._active_mice) == 0:
            raise RuntimeError('Animals must be specified before fitting.')
        elif self._active_key not in self._params.keys() or overwrite:
            self._params[self._active_key] = {}
            return self._fit(self.train_data, **kwargs)

    def _fit(self, data, **kwargs):
        raise NotImplementedError

    def predict(self, **kwargs):
        return self._predict(**kwargs)

    def _predict(self, **kwargs):
        raise NotImplementedError

    def add_reward_times(self, t1, t2):
        t = self._patch_model.times(t=t1, s=t2, interevent=False)
        r = (self._V0*np.arange(1, t.shape[0]+1)/self.REWARD_VOLUME).astype(np.int64)
        vals, ids = np.unique(r, return_index=True)
        ids = ids[vals > 0]
        return t[ids]

    def get_expected_reward_time(self, N, t=0.0):
        # Cache values since this function can be computationally expensive.
        key = (self._lambda0, self._tau, N, t)
        if key in self._E_reward_time.keys():
            return self._E_reward_time[key]

        # Get model functions (for shorthand)
        lam = self._patch_model.lam
        Lam = self._patch_model.Lam

        # Determine probability of Nth event ever occurring in a sequence
        Lam_inf = lambda t: self._lambda0*self._tau*np.exp(-t/self._tau) # Lam(t, inf)
        F0 = lambda t, n: 1.0 - np.exp(-Lam_inf(t))*np.sum(np.power(Lam_inf(t), np.arange(n))
                                                           /factorial(np.arange(n)))

        # Define s*p(s; t) to integrate. Note that leaving the sum inside the
        # integral speeds up computation.
        f = lambda s, t, n: s*np.sum((1.0/factorial(np.arange(n)))*np.exp(-Lam(t,s))*lam(t+s)
                                     *np.power(Lam(t,s), np.arange(n)-1)*(Lam(t,s) - np.arange(n)))

        # Need to pass points parameter since much of F is close to zero.
        # See https://stackoverflow.com/a/51053180
        pts = np.geomspace(0.01, 1000.0, num=25)

        # Integrate s*p(s; t) to find E[s]
        with np.errstate(over='raise'):
            # Attempt to integrate function as whole.
            try:
                # Integrate function
                dt = integrate.quad(f, 0.0, 1000.0,
                                    args=(t, N),
                                    points=pts)[0]

                # Divide by normalization constant. Handle division error
                # for F0 ~ 0.
                if F0(t, N) > 0.0:
                    dt /= F0(t, N)
                else:
                    dt = np.inf

            # If overflow from large N, integrate in segments. This is a
            # not exact but a fairly close approximation.
            except FloatingPointError:
                # Break number of events into segments
                K = 4 # number of segments
                N_seg = [N//K for k in range(K)]
                if N % K > 0:
                    N_seg += N % K
                
                # Iterate through each segment
                t_start = t # start of each segment
                for n in N_seg:
                    dt_k = integrate.quad(f, 0.0, 1000.0,
                                          args=(t_start, n),
                                          points=pts)[0]
                    if F0(t_start, n) > 0.0:
                        # Normalize if F0 finite within floating point precision.
                        dt_k /= F0(t_start, n)
                        t_start += dt_k
                    else:
                        # Otherwise, set to infinity.
                        t_start = np.inf
                        break
                
                # Save time from initial start to last event
                dt = t_start - t    
        
        # Limit to estimated maximum wait time
        if ('dt_mean' in self.params.keys()) and ('dt_std' in self.params.keys()):
            dt_max = self.params['dt_mean'] + 2*self.params['dt_std'] 
        else:
            dt_max = self.MAX_WAIT_TIME
        dt = np.minimum(dt, dt_max)
        
        # Cache value
        self._E_reward_time[key] = t + dt
        
        return t + dt


class HeuristicModel(ForagingModel):

    def __init__(self):
        super().__init__()


class RewardTimeModel(HeuristicModel):

    def __init__(self):
        super().__init__()

        # Set data
        self._data_keys += ['t_motor_patch', 'dt_patch']

        # Set params
        self._param_keys += ['dt_mean', 'dt_median']
        
    def _fit(self, data):
        # Get relevant data
        args = (data['t_motor_patch'], data['dt_patch'], data['env_id'])
        t_motor_patch, (dt_patch, env_ids) = self.unravel(*args)
        assert len(t_motor_patch) == len(dt_patch) == len(env_ids)

        # Compute all latencies between last reward and patch leaving
        dt_reward = []

        # Iterate over patches
        for t_mp, t_p, env_id in zip(t_motor_patch, dt_patch, env_ids):
            # Iterate over patches within session
            if t_mp.size > 0:
                # Compute delay between last reward and patch leaving.
                dt_reward.append(t_p - t_mp[-1])
            else:
                # If no rewards in patch, then take total time
                # (because never saw a reward).
                dt_reward.append(t_p)
        
        # Get appropriate measures of center of distributions
        self._params[self._active_key]['dt_mean']   = np.mean(dt_reward)
        self._params[self._active_key]['dt_std']    = np.std(dt_reward)
        self._params[self._active_key]['dt_median'] = np.median(dt_reward)

        return dt_reward

    def _predict(self, *, t_motor_patch=None, dt_patch=None, env_ids=None, method='mean'):
        # Get current parameters
        key = 'dt_' + method
        dt = self._params[self._active_key][key]

        # Get test data
        if t_motor_patch is None:
            t_motor_patch = self.test_data['t_motor_patch']
        if dt_patch is None:
            dt_patch = self.test_data['dt_patch']
        if env_ids is None:
            env_ids = self.test_data['env_id']

        # Unravel session data
        t_motor_patch, (dt_patch, env_ids,) = self.unravel(t_motor_patch, 
                                                           dt_patch, 
                                                           env_ids)
        assert len(t_motor_patch) == len(dt_patch) == len(env_ids)

        # Compute inter-reward intervals
        t_hat = np.ones([len(t_motor_patch)])*np.nan
        for j, (t_m, t_p, env_id) in enumerate(zip(t_motor_patch, dt_patch, env_ids)):
            if np.isnan(t_m).any():
                continue
            idx = np.atleast_1d(np.argwhere(np.diff(np.append(t_m, t_p)) > dt).squeeze())
            if len(idx) > 0:
                # Predict first elapsed time after reward > mean elapsed time
                t_hat[j] = t_m[idx[0]] + dt
            else:
                # Otherwise, estimate from simulated future reward times
                if len(t_m) == 0:
                    t_start = 0.0
                else:
                    t_start = t_m[-1]

                # Get number of drips per observable reward. Use int() instead
                # of floor division (//), which can yield floating point errors:
                # https://stackoverflow.com/a/38589899
                self.set_environment(env_id)
                N = int(self.REWARD_VOLUME/self._V0)

                # Determine expected waiting times for next rewards
                t = t_start
                iters = 0
                while (t - t_start) < dt:
                    t_start = t
                    t = self.get_expected_reward_time(N, t_start)
                    iters += 1
                    if iters > 50:
                        raise RuntimeError('No convergence.')
                t_hat[j] = t_start + dt
                #t = np.insert(self.add_reward_times(t_start, 1000.0), 0, t_start)
                #idx = np.append(np.argwhere(np.diff(t) > dt).squeeze(), -1)

                #t_hat[j] = t[idx[0]] + dt

        return t_hat

    def clear_cache(self):
        self._E_reward_time = {}


class RewardNumberModel(RewardTimeModel):

    def __init__(self):
        super().__init__()

        # Set data
        self._data_keys += ['n_rewards']

        # Set params
        self._param_keys += ['n_mean', 'n_median']

    def _fit(self, data):
        # Fit reward time model
        super()._fit(data)

        # Get relevant data
        n_rewards = self.unravel(data['n_rewards'])

        # Get appropriate measures of center of distributions
        self._params[self._active_key]['n_mean']   = np.mean(n_rewards)
        self._params[self._active_key]['n_median'] = np.median(n_rewards)

    def _predict(self, *, t_motor_patch=None, env_ids=None, method='mean'):
        # Get parameters
        dt = self._params[self._active_key]['dt_' + method]
        std = self._params[self._active_key]['dt_std']
        n = self._params[self._active_key]['n_' + method]

        # Get test data
        if t_motor_patch is None:
            t_motor_patch = self.test_data['t_motor_patch']
        if env_ids is None:
            env_ids = self.test_data['env_id']

        # Unravel session data
        t_motor_patch, (env_ids,) = self.unravel(t_motor_patch, env_ids)
        assert len(t_motor_patch) == len(env_ids)

        # Set random number generator with seed for reproducibility.
        rng = np.random.default_rng(self.RANDOM_SEED)

        # Compute inter-reward intervals
        t_hat = np.ones([len(t_motor_patch)])*np.nan
        for i, (t_m, env_id) in enumerate(zip(t_motor_patch, env_ids)):
            if np.isnan(t_m).any():
                continue

            # Determine reward threshold if fraction
            if rng.random() < (n%1):
                n_i = np.ceil(n).astype(np.int64)
            else:
                n_i = np.floor(n).astype(np.int64)

            # If fewer rewards then parameter for leaving, generate
            # new reward times using environment parameters.
            if len(t_m) < n_i:
                # Set environment
                self.set_environment(env_id)

                # Generate new reward times
                if len(t_m) == 0:
                   t_start = 0.0
                else:
                   t_start = t_m[-1]
                #t_m = np.insert(self.add_reward_times(t_start, 1000.0), 0, t_m)

                # If no reward times, set to 0.0.
                #if t_m.size == 0:
                #    t_m = np.array([0.0])

                # Find expected time of reward n_i
                N = (n_i - len(t_m))*int(self.REWARD_VOLUME/self._V0)
                t_hat[i] = self.get_expected_reward_time(N, t_start)


            # Otherwise, take time of nth reward
            else:
                t_hat[i] = t_m[n_i-1]
            
            # if len(t_m) < n_i:
            #     # If still fewer rewards than parameter, estimate leaving
            #     # time as two std above mean delay.
            #     t_hat[i] = t_m[-1] + dt + 2*std
            # else:
            #     # Otherwise, base leaving time on nth reward time.
            #     t_hat[i] = t_m[n_i-1] + dt

        return t_hat

class TimeOnTaskModel(HeuristicModel):

    MODEL_TYPES = ['linear', 'log-linear']

    def __init__(self, model_type):
        super().__init__()

        # Set data
        self._data_keys += ['n_patch', 't_patch', 'dt_patch']

        # Set params
        self._param_keys += ['model']

        # Save regression model type
        if model_type not in self.MODEL_TYPES:
            raise ValueError('Model type {} not known.'.format(model_type))
        self.model_type = model_type

    def _fit(self, data, metric='n_patch'):
        # Check time-on-task metric
        if metric not in ['n_patch', 't_patch']:
            raise ValueError('Time-on-task metric {} not known.'.format(metric))
        elif metric not in data.keys():
            raise KeyError('Time-on-task metric {} not found in dataset.'.format(metric))
        self.metric = metric

        # Get relevant data
        tot = np.array(self.unravel(data[metric])) # patch number or time
        dt_patch = np.array(self.unravel(data['dt_patch'])) # residence times
        if self.model_type == 'log-linear':
            dt_patch = np.log(dt_patch)

        # Fit regression model
        model = ephys.LinearRegression(use_bias=True)
        model.fit(tot, dt_patch)
        self._params[self._active_key]['model'] = model

    def _predict(self, *, n_patch=None, t_patch=None):
        # Set time-on-task values
        X = None
        if (self.metric == 'n_patch') and (n_patch is not None):
            X = n_patch
        elif (self.metric == 't_patch') and (t_patch is not None):
            X = t_patch
        else:
            X = self.test_data[self.metric]

        # Format data
        X = np.array(self.unravel(X))
        
        # Predict residence times from regression model
        t_hat = self._params[self._active_key]['model'].predict(X)
        if self.model_type == 'log-linear':
            return np.atleast_1d(np.exp(t_hat).squeeze())
        else:
            return np.atleast_1d(t_hat.squeeze())

class MVTModel(ForagingModel):

    def __init__(self):
        super().__init__()

        # Set data
        self._data_keys += ['dt_interpatch', 'dt_patch']

        # Set params
        self._param_keys += ['dt_est']

    def _fit(self, data, percentile=0.1):
        # Check environment
        #if len(self._active_envs) > 1:
        #    raise RuntimeError('Only one environment can be active during model'
        #                       ' fitting or inference.')
        #self.set_environment(self._active_envs[0])

        # Estimate true travel time
        dt_interpatch = np.array(self.unravel(data['dt_interpatch']))
        idx = int(len(dt_interpatch)*percentile)
        if len(dt_interpatch) == 0:
            self._params[self._active_key]['dt_est'] = np.array([])
        else:
            self._params[self._active_key]['dt_est'] = np.sort(dt_interpatch)[idx]

    def _predict(self, *, env_ids=None):
        # Check environment
        #if len(self._active_envs) > 1:
        #    raise RuntimeError('Only one environment can be active during model'
        #                       ' fitting or inference.')
        #self.set_environment(self._active_envs[0])

        # Get relevant data
        if env_ids is None:
            # Use other data to broadcast env_ids appropriately
            dt_interpatch = self.test_data['dt_interpatch']
            env_ids = self.test_data['env_id']
            dt_interpatch, (env_ids,) = self.unravel(dt_interpatch, env_ids)
            env_ids = np.array(env_ids)
        elif isinstance(env_ids, list):
            env_ids = np.array(self.unravel(env_ids))

        # Compute optimal residence times
        t_hat = np.ones([len(env_ids)])*np.nan
        for env_id in np.unique(env_ids):
            # Check environment
            if env_id not in self._active_envs:
                raise RuntimeError('Environments for prediction must be in active'
                                   ' list. Use set_data() to add environment {}.'
                                   .format(env_id))
            
            # Get environment parmeters
            self.set_environment(env_id)
            r_0 = self._V0*self._lambda0
            tau = self._tau
            t_t = self._params[self._active_key]['dt_est']

            # Find optimal residence times
            preds, _ = helper.get_optimal_values(t_t=t_t, 
                                                 R_0=0.0, 
                                                 r_0=r_0, 
                                                 tau=tau, 
                                                 multipatch=False)
            idx, = np.nonzero(env_ids == env_id)
            t_hat[idx] = np.atleast_1d(preds.squeeze())
        
        return t_hat

class FittedMVTModel(MVTModel):

    PARAM_NAMES = ['tau', 't_t'] # params to fit
    MVT_PARAMS = ['t_t', 'track', 'R_0', 'r_0', 'tau'] # params for MVT equation

    def __init__(self):
        super().__init__()

        # Set params
        self._param_keys += [n + '_exp' for n in self.PARAM_NAMES]
        self._param_keys += [n + '_fit' for n in self.PARAM_NAMES]
        self._param_keys += ['logerr']

    def _fit(self, 
             data, 
             param_names,
             constrain=False,
             percentile=0.1):
        # Check params
        if not isinstance(param_names, list):
            param_names = [param_names]
        if any([name not in self.PARAM_NAMES for name in param_names]):
            raise ValueError('Parameter name not recognized.')

        # Get relevant data
        env_ids = self._active_envs

        # Get experimental parameter and residence time values for each condition.
        # The true residences times will be a list of arrays, where each array
        # represents all residence times for one environment. Parameters will
        # be dictionaries consisting of parameter values and indices to map into
        # the error function so that it can be fed into the scipy.optimize 
        # framework.
        Y = []
        params = {k: {'value': np.zeros([len(env_ids)])} for k in self.MVT_PARAMS}
        
        # Add set parameter values for environment
        for i, env_id in enumerate(env_ids):
            # Get parameter values
            params['track']['value'][i] = self.env_params[env_id]['track']
            params['r_0']['value'][i] = self.env_params[env_id]['V0'] \
                                        *self.env_params[env_id]['lambda0']
            params['tau']['value'][i] = self.env_params[env_id]['tau']

            # Get training data
            self.set_data(mouse_ids=self._active_mice, 
                          env_ids=env_id,
                          k=self._k,
                          index=self._test_index)
            Y.append(np.array(self.unravel(self.train_data['dt_patch'])))
        
        # Add estimated travel time for each subset of environments with the
        # same track length.
        for val in np.unique(params['track']['value']):
            # Set and fit to data subset.
            envs = [env_id for env_id in env_ids
                    if self.env_params[env_id]['track'] == val]
            self.set_data(mouse_ids=self._active_mice, 
                            env_ids=envs,
                            k=self._k,
                            index=self._test_index)
            if self._active_key not in self._params.keys():
                self._params[self._active_key] = {}
            super()._fit(self.train_data)

            # Assign values in corresponding parameter placeholder
            idx = np.array([env_ids.index(env) for env in envs])
            params['t_t']['value'][idx] = self._params[self._active_key]['dt_est']

        if constrain:
            # Build initial vector by concatenating initial experimental values
            # for each parameter.
            x_init = []
            n = 0
            for var in param_names:
                # Get unique values and indices of parameter over subset of conditions,
                # where vals is unique values, and ids is indices that transform vector
                # of initial values back to vector of parameter values for all conditions.
                valset, ids = np.unique(params[var]['value'], return_inverse=True)

                # Save unique values, indices of parameter unique values in overall
                # concatenated vector in minimize(), and indices to recover initial
                # vector of all parameter values.
                x_init.append(valset)
                params[var]['index'] = slice(n, n + len(valset))
                params[var]['ids'] = ids
                n += len(valset)
            
            # Concatenate into single vector for minimize()
            x_init = np.hstack(x_init)
        
        else:
            # Assign single unique value for each test parameter
            x_init = np.zeros([len(param_names)])
            for i, var in enumerate(param_names):
                # Check that environment parameters are consistent
                if not all([self.env_params[env_id][var] == self.env_params[env_ids[0]][var]
                            for env_id in env_ids]):
                    raise RuntimeError('Unconstrained fitting requires environment'
                                       ' parameters to be consistent but {} has'
                                       ' different values.'.format(var))
                
                # Assign 
                x_init[i] = params[var]['value'][0]
                params[var]['index'] = i
                params[var]['ids'] = ()
                
        # Reset data (to reset active key)
        self.set_data(mouse_ids=self._active_mice, 
                      env_ids=env_ids,
                      k=self._k,
                      index=self._test_index)

        # Save experimental values (also passed as initial guess in x_init)
        self._params[self._active_key]['tau_exp'] = np.unique(params['tau']['value'])
        vals, idx = np.unique(params['t_t']['value'], return_index=True)
        self._params[self._active_key]['t_t_exp'] = vals
        self._params[self._active_key]['track_exp'] = params['track']['value'][idx]
                        
        # Optimize x as minimizing MSE of residence times
        res = opt.minimize(self.mvt_error, x_init, args=(Y, params))

        # Save optimization results
        self._params[self._active_key]['names'] = param_names
        if res.success:
            for var in param_names:
                self._params[self._active_key][var + '_fit'] = \
                    np.atleast_1d(res.x[params[var]['index']])
            self._params[self._active_key]['logerr'] = res.fun
        else:
            for var in param_names:
                self._params[self._active_key][var + '_fit'] = \
                    np.array([np.nan])
            self._params[self._active_key]['logerr'] = np.nan

    @staticmethod
    def mvt_error(x, Y, params, broadcast_shape=False):
        # Set parameter values
        loc = []
        for var in ['t_t', 'R_0', 'r_0', 'tau']:
            idx = params[var].get('index', None)
            if idx is None:
                # If index not provided, pass value of parameter in each condition.
                loc.append(params[var]['value'])
            else:
                # Use index to index x for unique values, and then use ids to expand
                # vector in order to assign value for each condition.
                loc.append(x[params[var]['index']][params[var]['ids']])
                
        t_t, R_0, r_0, tau = loc 
        
        # Get optimal residence times from parameters
        y_hat, _ = helper.get_optimal_values(t_p=None, 
                                            t_t=t_t,
                                            R_0=R_0, 
                                            r_0=r_0, 
                                            tau=tau,
                                            broadcast_shape=broadcast_shape)
        y_hat = np.atleast_1d(y_hat.squeeze())
        
        # Compute error among condition sets
        error = []
        for i in range(len(Y)):
            error.append((Y[i] - y_hat[i])**2)
        
        # Return logsum of squared error (more numerically stable)
        return np.log(np.sum(np.hstack(error)))

    def _predict(self, *, env_ids=None):
        # Get relevant data
        if env_ids is None:
            # Use other data to broadcast env_ids appropriately
            dt_interpatch = self.test_data['dt_interpatch']
            env_ids = self.test_data['env_id']
            dt_interpatch, (env_ids,) = self.unravel(dt_interpatch, env_ids)
            env_ids = np.array(env_ids)
        elif isinstance(env_ids, list):
            env_ids = np.array(self.unravel(env_ids))

        # Compute optimal residence times
        t_hat = np.ones([len(env_ids)])*np.nan
        for env_id in np.unique(env_ids):
            # Check environment
            if env_id not in self._active_envs:
                raise RuntimeError('Environments for prediction must be in active'
                                   ' list. Use set_data() to add environment {}.'
                                   .format(env_id))
            
            # Get experimental and fitted environment parmeters
            self.set_environment(env_id)
            r_0 = self._V0*self._lambda0
            if 'tau' in self._params[self._active_key]['names']:
                idx = (self._params[self._active_key]['tau_exp'] == self._tau)
                tau = self._params[self._active_key]['tau_fit'][idx]
            else:
                tau = self._tau
            idx = (self._params[self._active_key]['track_exp'] == self._track)
            if 't_t' in self._params[self._active_key]['names']:
                t_t = self._params[self._active_key]['t_t_fit'][idx]
            else:
                t_t = self._params[self._active_key]['t_t_exp'][idx]
            
            # Find optimal residence times
            preds, _ = helper.get_optimal_values(t_t=t_t, 
                                                 R_0=0.0, 
                                                 r_0=r_0, 
                                                 tau=tau, 
                                                 multipatch=False)
            idx, = np.nonzero(env_ids == env_id)
            t_hat[idx] = np.atleast_1d(preds.squeeze())
        
        return t_hat
