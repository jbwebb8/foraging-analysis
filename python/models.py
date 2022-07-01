import copy
import itertools, functools
import warnings
import numpy as np
import scipy.optimize as opt
from scipy import integrate
from scipy.special import factorial
import matplotlib.pyplot as plt
import ephys
import util
import helper

warnings.filterwarnings('default',
                        'Patch numbers must be zero-indexed.',
                        category=SyntaxWarning,
                        module='models')
warnings.filterwarnings('default',
                        'Environment failed to fit properly. Using '
                        'experimental values for (tau|t_t) instead.',
                        category=RuntimeWarning,
                        module='models')
warnings.filterwarnings('default',
                        'Range of tau values not specified. Using default values:*',
                        category=RuntimeWarning,
                        module='models')

                        

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
        # Model name
        self.name = 'base'

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
        self.index_data = False # data subsets consist of all data plus train/test indices

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
        return self._get_subset_data(dataset='train', indexed=self.index_data)

    @property
    def test_data(self):
        return self._get_subset_data(dataset='test', indexed=self.index_data)

    @property
    def all_data(self):
        return self._get_subset_data(dataset='all', indexed=False)

    def _get_subset_data(self, dataset, indexed=False):
        # Get relevant indices
        if dataset == 'train':
            compare = lambda i, j: i != j
        elif dataset == 'test':
            compare = lambda i, j: i == j
        elif dataset == 'all':
            compare = lambda i, j: True
        else:
            raise ValueError('Dataset {} not recognized.'.format(dataset))

        # Check for proper initialization.
        if len(self._data_keys) == 0:
            raise KeyError('Data keys not initialized.')
        if self._k < 0:
            raise RuntimeError('Data index not initialized. Use set_data() to'
                               + ' initialize data subsets.')
        
        # Format 1: data[<data_key>] = list of pooled session data
        # Add data if 1) involves active mice and environment, and 2) is part of the
        # queried data subset (i.e. train or test).
        data = {k: [] for k in self._data_keys + ['mouse_id', 'env_id'] + ['index']*indexed}
        for key in itertools.product(self._active_mice, self._active_envs):
            if key in self._data.keys():
                # Iterate over sessions
                for i, subset_idx in enumerate(self._subset_idx[key]):
                    # Append data corresponding to appropriate indices within session
                    idx = np.hstack([ids for j, ids in enumerate(subset_idx) 
                                     if compare(j, self._test_index)])
                    if indexed:
                        # Provide boolean indexing to data subset, and reset other
                        # indices to include all data.
                        idx_all = np.hstack(subset_idx)
                        data['index'].append(np.isin(idx_all, idx))
                        idx = idx_all # reset indices to all
                    for k in self._data_keys:
                        # data[(mouse_id, env_id)][<data_name>][<session>][<subset>]
                        if isinstance(self._data[key][k][i], np.ndarray):
                            data[k].append(self._data[key][k][i][idx])
                        else:
                            data[k].append([self._data[key][k][i][n] for n in idx])
                    
                    # Add animal and environment IDs associated with session
                    data['mouse_id'].append(key[0])
                    data['env_id'].append(key[1])

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
            missing_ids = [str(m) for m in mouse_ids if m not in self._mice]
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
            missing_ids = [str(m) for m in env_ids if m not in self._envs]
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
                            idx[idx_rem] = np.append(idx[idx_rem], rand_idx[-(j+1)])
                        
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

    def cross_validate(self, mouse_ids, 
                       env_ids, K, 
                       verbose=True, 
                       fit_kwargs={}, 
                       pred_kwargs={},
                       return_data=[]):
        if verbose:
            print('Analyzing animal(s) {}... k = '.format(mouse_ids), end='')
        
        # Check inputs.
        if not isinstance(return_data, list):
            return_data = [return_data]

        # Get current dataset parameters.
        cache = {'mouse_ids': self._active_mice, 
                 'env_ids': self._active_envs,
                 'k': self._k,
                 'index': getattr(self, '_test_index', None)}
                 
        # Placeholders
        Y = []
        Y_hat = []
        ret = [[] for name in return_data]

        # Iterate over cross-validation indices.
        for k in range(K):
            if verbose:
                print('{}'.format(k), end=' ')
            
            # Fit to training subset
            self.set_data(mouse_ids=mouse_ids, env_ids=env_ids, k=K, index=k)
            self.fit(**fit_kwargs)

            # Get test data and predictions.
            y = self.unravel(self.test_data['dt_patch'],
                             *[self.test_data[name] for name in return_data])
            if len(return_data) > 0:
                y, args = y
                for i, arg in enumerate(args):
                    ret[i].append(np.array(arg))
            y = np.array(y)
            if self.index_data:
                _, (index,) = self.unravel(self.test_data['dt_patch'],
                                           self.test_data['index'])
                y = y[index]
                ret = [[a[index] for a in r] for r in ret]
            Y.append(y)
            Y_hat.append(self.predict(**pred_kwargs))
        
        # Reset dataset to original if specified.
        if cache['k'] > -1:
            self.set_data(**cache)

        if verbose:
            print('done.')

        if len(return_data) > 0:
            return Y, Y_hat, tuple(ret)
        else:
            return Y, Y_hat

    def add_reward_times(self, t1, t2):
        t = self._patch_model.times(t=t1, s=t2, interevent=False)
        r = (self._V0*np.arange(1, t.shape[0]+1)/self.REWARD_VOLUME).astype(np.int64)
        vals, ids = np.unique(r, return_index=True)
        ids = ids[vals > 0]
        return t[ids]

    def get_expected_reward_time(self, N, t=0.0, epsilon=0.01):
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
                if F0(t, N) > epsilon:
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
        
        # Cache value
        self._E_reward_time[key] = (t + dt, F0(t, N))

        return t + dt, F0(t, N)

    def get_expected_wait_time(self, N, t=0.0, N0=0):
        # Check validity of inputs.
        if N0 > N:
            raise ValueError('Number of unobserved events must not exceed'
                             ' observation threshold.')

        # Find expected time to next reward
        tN, F0 = self.get_expected_reward_time(N-N0, t)
        dt = tN - t

        # Limit to estimated maximum wait time
        if ('dt_mean' in self.params.keys()) and ('dt_std' in self.params.keys()):
            dt_max = self.params['dt_mean'] + 2*self.params['dt_std'] 
        else:
            dt_max = self.MAX_WAIT_TIME
        dt = np.minimum(dt, dt_max)

        return t + dt

    def add_expected_reward_times(self, N, t, T_max, N0=0, max_iters=50):
        # Check validity of inputs.
        if N0 > N:
            raise ValueError('Number of unobserved events must not exceed'
                             ' observation threshold.')
        
        # Determine expected waiting times for next rewards
        i = 0 # number of iterations
        t_new = [] # list of new reward times
        keep = [] # keep new candidate time if True
        rng = np.random.default_rng(self.RANDOM_SEED) # rng for comparing F0
        while t < T_max:
            # Get reward time based on previous expected time. While this approach
            # is not as precise as determining the expected time of reward
            # numbers that are multiples of N from the same start time, it is
            # more numerically stable, since the intermediate values in the
            # computation scale much more quickly with N than t.
            if i == 0:
                # If first reward time, incorporate unobserved events.
                t, F0 = self.get_expected_reward_time(N-N0, t)
            else:
                # Otherwise, find time to N events.
                t, F0 = self.get_expected_reward_time(N, t)
            t_new.append(t)
            keep.append(rng.random() < F0)
            i += 1
            if i > 50:
                raise RuntimeError('No convergence.')
        
        # Sanity check for sensible values. Time between expected rewards should
        # increase with time (i.e. second derivative always positive).
        t_new = np.array(t_new)
        if (np.diff(np.diff(t_new)) < 0.0).any():
            raise RuntimeError('Decreasing expected reward times detected in output.')

        # Prune new candidate reward times by probability of occurrence.
        t_new = t_new[np.array(keep)]

        return t_new

    def estimate_unobserved_events(self, N, t, s):
        # Get model functions (for shorthand)
        lam = self._patch_model.lam
        Lam = self._patch_model.Lam

        # Compute expectation of number of unobserved events:
        #   E[n] = sum(n*p(n))/sum(p(n))
        # where summation is for n = 0,...,N-1
        p = lambda s, t, n: np.exp(-Lam(t,s))*(Lam(t,s)**n)/factorial(n)
        E_n = np.sum(np.arange(N)*p(s, t, np.arange(N)))/np.sum(p(s, t, np.arange(N)))

        # Round to nearest integer (except down if > N-1).
        E_n = min(round(E_n), N-1)

        return E_n

    def _build_opt_inputs(self, 
                          param_names,
                          param_inits,
                          constrain=False):
        # Get experimental parameter and residence time values for each condition.
        # The true residences times will be a list of arrays, where each array
        # represents all residence times for one environment. Parameters will
        # be dictionaries consisting of parameter values and indices to map into
        # the error function so that it can be fed into the scipy.optimize 
        # framework.
        params = {k: {'value': param_inits[k]} for k in param_inits.keys()}
        
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
                x_init[i] = params[var]['value'][0]
                params[var]['index'] = i
                params[var]['ids'] = ()

        return x_init, params

    def save(self, filename):
        f = open(filepath, 'wb')
        d = self.__dict__.copy()
        _ = d.pop('_patch_model', None) # cannot pickle due to namedtuple
        pickle.dump(d, f)
        f.close()

    def load(self, filename):
        f = open(filepath, 'rb')
        d = pickle.load(f)
        for k, v in d.items():
            model.__dict__[k] = v
        f.close()


class HeuristicModel(ForagingModel):

    def __init__(self):
        """Placeholder class for any future updates to heuristic models."""
        super().__init__()
        self.name = 'heuristic'


class FixedTimeModel(HeuristicModel):
    def __init__(self):
        super().__init__()
        self.name = 'dt_fixed'

        # Set data
        self._data_keys += ['dt_patch']

        # Set params
        self._param_keys += ['dt_mean', 'dt_median']

    def _fit(self, data):
        # Get relevant data
        args = (data['dt_patch'], data['env_id'])
        dt_patch, (env_ids,) = self.unravel(*args)
        assert len(dt_patch) == len(env_ids)
        
        # Get appropriate measures of center of distributions
        self._params[self._active_key]['dt_mean']   = np.mean(dt_patch)
        self._params[self._active_key]['dt_std']    = np.std(dt_patch)
        self._params[self._active_key]['dt_median'] = np.median(dt_patch)

        return dt_patch

    def _predict(self, *, dt_patch=None, env_ids=None, method='mean'):
        # Get current parameters
        key = 'dt_' + method
        dt = self._params[self._active_key][key]

        # Get test data
        if dt_patch is None:
            dt_patch = self.test_data['dt_patch']
        if env_ids is None:
            env_ids = self.test_data['env_id']

        # Unravel session data
        dt_patch, (env_ids,) = self.unravel(dt_patch, env_ids)
        assert len(dt_patch) == len(env_ids)

        # Predict as single estimate
        t_hat = np.ones([len(dt_patch)])*dt

        return t_hat


class RewardTimeModel(HeuristicModel):

    def __init__(self):
        super().__init__()
        self.name = 'dt_reward'

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
            # TODO: does this filter out properly?
            if np.isnan(t_m).any():
                continue
            
            # Determine which inter-reward intervals satisfy criteria
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

                # Estimate number of unobserved drips at leaving time.
                N0 = self.estimate_unobserved_events(N, t_start, t_p-t_start)

                # Determine expected waiting times for next rewards
                t = t_start
                iters = 0
                while (t - t_start) < dt:
                    t_start = t
                    if iters == 0:
                        t = self.get_expected_wait_time(N-N0, t_p)
                    else:
                        t = self.get_expected_wait_time(N, t_start)
                    iters += 1
                    if iters > 50:
                        raise RuntimeError('No convergence.')
                t_hat[j] = t_start + dt

        return t_hat

    def clear_cache(self):
        self._E_reward_time = {}


class RewardNumberModel(RewardTimeModel):

    def __init__(self):
        super().__init__()
        self.name = 'n_reward'

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

        return n_rewards

    def _predict(self, *, 
                 t_motor_patch=None, 
                 dt_patch=None, 
                 env_ids=None, 
                 method='mean',
                 delay=None):
        # Get parameters
        n = self._params[self._active_key]['n_' + method]
        if delay is None:
            dt = 0.0
        elif isinstance(delay, str):
            dt = self._params[self._active_key]['dt_' + delay]
        else:
            dt = float(delay)

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

        # Set random number generator with seed for reproducibility.
        rng = np.random.default_rng(self.RANDOM_SEED)

        # Compute inter-reward intervals
        t_hat = np.ones([len(t_motor_patch)])*np.nan
        for i, (t_m, t_p, env_id) in enumerate(zip(t_motor_patch, dt_patch, env_ids)):
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

                # Find expected time of reward n_i
                N = (n_i - len(t_m))*int(self.REWARD_VOLUME/self._V0)
                try:
                    N0 = self.estimate_unobserved_events(N, t_start, t_p-t_start)
                except ValueError:
                    print(N, t_start, t_p, env_id, n_i, t_m)
                t_hat[i] = self.get_expected_wait_time(N-N0, t_p) + dt

            # Otherwise, take time of nth reward
            else:
                t_hat[i] = t_m[n_i-1] + dt

        return t_hat

class TimeOnTaskModel(HeuristicModel):

    MODEL_TYPES = ['linear', 'log-linear']

    def __init__(self, model_type):
        super().__init__()
        self.name = 'tot'

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
        self.name = 'mvt'

        # Set data
        self._data_keys += ['dt_interpatch', 'dt_patch']

        # Set params
        self._param_keys += ['dt_est']

    def _fit(self, data, percentile=0.1):
        # Get relevant data
        args = (data['dt_interpatch'], data['env_id'])
        dt_interpatch, (env_ids,) = self.unravel(*args)
        dt_interpatch = np.array(dt_interpatch)
        env_ids = np.array(env_ids)
        assert len(dt_interpatch) == len(env_ids)
        
        # Determine all track types in dataset.
        tracks = np.ones([len(env_ids)])*np.nan
        for env_id in np.unique(env_ids):
            idx = (env_ids == env_id)
            tracks[idx] = self.env_params[env_id]['track']
        
        # Estimate true travel time for each track length.
        if 'dt_est' not in self._params[self._active_key].keys():
            self._params[self._active_key]['dt_est'] = {}
        for track in np.unique(tracks):
            idx = (tracks == track)
            index = int((len(dt_interpatch[idx])-1)*percentile)
            if len(dt_interpatch) == 0:
                self._params[self._active_key]['dt_est'][track] = np.array([])
            else:
                self._params[self._active_key]['dt_est'][track] = np.sort(dt_interpatch[idx])[index]

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
            elif self.env_params[env_id]['track'] not in self.params['dt_est'].keys():
                raise ValueError('Model has not been fit to track type {}.'
                .format(self.env_params[env_id]['track']))
            
            # Get environment parmeters
            self.set_environment(env_id)
            r_0 = self._V0*self._lambda0
            tau = self._tau
            t_t = self._params[self._active_key]['dt_est'][self._track]

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
        self.name = 'mvt_fit'

        # Set params
        self._param_keys += [n + '_exp' for n in self.PARAM_NAMES]
        self._param_keys += [n + '_fit' for n in self.PARAM_NAMES]
        self._param_keys += ['logerr']

    def _fit(self, 
             data, 
             param_names,
             constrain=False,
             plot_results=False,
             percentile=0.1,
             lam=0.0):
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
            super()._fit(self.train_data, percentile=percentile)

            # Assign values in corresponding parameter placeholder
            idx = np.array([env_ids.index(env) for env in envs])
            params['t_t']['value'][idx] = self._params[self._active_key]['dt_est'][val]

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
            name_map = {'t_t': 'track', 'tau': 'tau'}
            for i, var in enumerate(param_names):
                # Check that environment parameters are consistent
                if not all([self.env_params[env_id][name_map[var]] == self.env_params[env_ids[0]][name_map[var]]
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
        exp_params = {name: params[name]['value'] for name in self.MVT_PARAMS}
                        
        # Optimize x as minimizing MSE of residence times
        for gtol in [1e-5, 1e-4, 1e-3]:
            res = opt.minimize(self.mvt_error, x_init, 
                               args=(Y, params, exp_params, lam), 
                               method='BFGS',
                               options={'gtol': gtol})
            if res.success:
                break

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

        # Visualize results for unconstrained fits
        if plot_results and not constrain:
            ax = plt.gca()
            cmap = plt.get_cmap('bone')
            if len(param_names) == 1:
                x = np.linspace(1, 2*res.x, num=50)
                err = np.zeros([50])
                for j, x_j in enumerate(x):
                    err[j] = self.mvt_error(x_j, Y, params, exp_params, lam)
                ax.plot(x, err, color=cmap(0.5))
                ax.axvline(res.x, color=cmap(0.1), linestyle='--')
                ax.set_xlabel(param_names[0])
                ax.set_ylabel('log(error)')
            elif len(param_names) == 2:
                #x1, x2 = np.meshgrid(x[:,0], x[:,1])
                x = [np.linspace(1, 2*res.x[params[var]['index']], num=50) 
                     for var in param_names]
                x1, x2 = np.meshgrid(x[0], x[1])
                err = np.zeros([50, 50])
                for j in range(50):
                    for k in range(50):
                        err[j, k] = self.mvt_error(np.array([x1[j,k], x2[j,k]]), 
                                                   Y, params, exp_params, lam)
                dx = np.diff(x, axis=0).mean()
                ax.pcolormesh(x1, x2, err[:-1, :-1], cmap='coolwarm')
                ax.set_xlabel(param_names[0])
                ax.set_ylabel(param_names[1])
            title = '\n'.join(['{}={}'.format(var, self.params['{}_exp'.format(var)]) 
                               for var in param_names])
            ax.set_title('experimental params:\n' + title)

        return res, x_init, Y, params, exp_params

    @staticmethod
    def mvt_error(x, Y, params, exp_params, lam=0.0, broadcast_shape=False):
        # Check arguments
        assert lam >= 0.0, 'Regularization constant must be non-negative.'
        
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
        error = np.hstack(error)

        # Add regularization term
        reg = np.sum((t_t - exp_params['t_t'])**2 + (tau - exp_params['tau'])**2)
        
        # Return logsum of squared error (more numerically stable)
        return np.log(np.mean(error) + lam*reg)

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
                if not np.isnan(self._params[self._active_key]['tau_fit']).all():
                    tau = self._params[self._active_key]['tau_fit'][idx]
                else:
                    warnings.warn('Environment failed to fit properly. Using '
                                  'experimental values for tau instead.',
                                  category=RuntimeWarning)
                    tau = self._tau
            else:
                tau = self._tau
            idx = (self._params[self._active_key]['track_exp'] == self._track)
            if 't_t' in self._params[self._active_key]['names']:
                if not np.isnan(self._params[self._active_key]['t_t_fit']).all():
                    t_t = self._params[self._active_key]['t_t_fit'][idx]
                else:
                    warnings.warn('Environment failed to fit properly. Using '
                                  'experimental values for t_t instead.',
                                  category=RuntimeWarning)
                    t_t = self._params[self._active_key]['t_t_exp'][idx]
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


class BayesianModel(ForagingModel):

    DEFAULT_TAU_RANGE = np.append(np.geomspace(0.1, 1000.0, num=100), np.inf)

    def __init__(self, N):
        super().__init__()
        self.name = 'bayes'

        # Set data
        self._data_keys += ['t_motor_patch', 'dt_patch', 'n_patch', 'dt_interpatch']

        # Set fitted params
        self._param_keys += ['dt_est']
        self._param_keys += ['lam_thresh', 'll_thresh', 'lhw_thresh']
        self._param_keys += ['p_lam0', 'p_tau'] # priors

        # Set model parameters
        self._N = N # number of previous patches to use

        # Format data subsets
        self.index_data = True

    def _fit(self, data, *,
             ll_thresh=None,
             lhw_thresh=None,
             lam_thresh=None,
             p_lam0=None,
             p_tau=None,
             percentile=0.1):
        """
        Following the MVTModel class, fit only the estimated travel time.
        The decision parameters (ll_thresh and lam_thresh) are determined
        at initialization and prediction, respectively.

        EDIT: Here, we will fit the estimated travel time and subsequently
        MVT-optimal Poisson rate threshold for leaving (lam_thresh), while
        assigning the log-likelihood threshold for leaving (ll_thresh) based
        on the value at initialization. Note that this model requires all MVT
        parameters in the active environments (tau, track length) to be equal.

        EDIT: Now, thresholds are provided in the fit function exclusively. Log-
        likelihood and likelihood-width thresholds must be given, whereas the
        Poisson-rate threshold may either be given or fit to optimal MVT time 
        if lam_thresh is None. Further, each parameter may be either a single value
        for all environments or a dictionary of environment-value pairs.
        """
        # Ensure all MVT parameters equal in active environments if
        # Poisson-rate threshold will be fit.
        tau_equal = all([self.env_params[env_id]['tau'] == self.env_params[self._active_envs[0]]['tau'] 
                          for env_id in self._active_envs])
        track_equal = all([self.env_params[env_id]['track'] == self.env_params[self._active_envs[0]]['track'] 
                           for env_id in self._active_envs])
        if (lam_thresh is None) and not (track_equal and tau_equal):
           warnings.warn('Active environments have different MVT parameters.',
                         category=SyntaxWarning)

        # Convert to dict if needed.
        if not isinstance(ll_thresh, dict):
            ll_thresh = {env_id: ll_thresh for env_id in self._active_envs}
        if not isinstance(lhw_thresh, dict):
            lhw_thresh = {env_id: lhw_thresh for env_id in self._active_envs}
        if (not isinstance(lam_thresh, dict)) and (lam_thresh is not None):
            lam_thresh = {env_id: lam_thresh for env_id in self._active_envs}

        # Estimate true travel time
        dt_interpatch, (index,) = self.unravel(data['dt_interpatch'], data['index'])
        dt_interpatch = np.array(dt_interpatch)[index]
        idx = int((len(dt_interpatch)-1)*percentile)
        if len(dt_interpatch) == 0:
            self._params[self._active_key]['dt_est'] = np.array([])
            return # no data to fit
        else:
            self._params[self._active_key]['dt_est'] = np.sort(dt_interpatch)[idx]

        # Find thresholds for each environment
        self._params[self._active_key]['lam_thresh'] = {}
        self._params[self._active_key]['ll_thresh'] = {} # keep formatting consistent
        self._params[self._active_key]['lhw_thresh'] = {}
        for env_id in self._active_envs:
            if lam_thresh is None:
                # Get environment parameters
                self.set_environment(env_id)
                r_0 = self._V0*self._lambda0
                tau = self._tau
                t_t = self._params[self._active_key]['dt_est']

                # Find Poisson rate at optimal residence time
                t_opt, _ = helper.get_optimal_values(t_t=t_t, 
                                                    R_0=0.0, 
                                                    r_0=r_0, 
                                                    tau=tau, 
                                                    multipatch=False)
                lam = self._patch_model.lam(t_opt.squeeze())
            else:
                lam = lam_thresh[env_id]

            # Add threshold to model parameters
            self._params[self._active_key]['lam_thresh'][env_id] = lam
            self._params[self._active_key]['ll_thresh'][env_id] = ll_thresh[env_id]
            self._params[self._active_key]['lhw_thresh'][env_id] = lhw_thresh[env_id]

        # Add priors.
        if p_lam0 is None:
            p_lam0 = (1, 0)
        if (not isinstance(p_lam0, dict)):
            p_lam0 = {env_id: p_lam0 for env_id in self._active_envs}
        self._params[self._active_key]['p_lam0'] = p_lam0
        if p_tau is None:
            p_tau = (1, 0)
        if (not isinstance(p_tau, dict)):
            p_tau = {env_id: p_tau for env_id in self._active_envs}
        self._params[self._active_key]['p_tau'] = p_tau

    def _get_patch_indices(self, n_patch, index, error=True, alt=False):
        # Get indices corresponding to patches to analyze.
        n_model = np.arange(max(n_patch[index]-(1+int(alt))*self._N, 0), 
                            n_patch[index]+1,
                            1+int(alt))
        idx = np.argwhere(n_patch[np.newaxis, :] == n_model[:, np.newaxis])
        if idx.shape[0] < n_model.shape[0]:
            if error:
                raise SyntaxError('Insufficient data provided for {} previous patches.'
                                .format(self._N))
            else:
                return None
        else:
            return idx[:, 1]

    def _predict(self, *,
                 t_motor_patch=None,
                 dt_patch=None,
                 n_patch=None,
                 index=None,
                 env_ids=None,
                 bin_size=0.5,
                 T_max=60.0,
                 tau_range=None,
                 filt=None):
        """
        Find time at which lam_hat falls below threshold lam_thresh, where
        lam_thresh is the Poisson rate at optimal leaving time, given that
        the time-normalized log-likelihood for lam_hat exceeds some threshold
        ll_thresh.
        """
        # Get relevant values
        if t_motor_patch is None:
            t_motor_patch = self.test_data['t_motor_patch']
        if dt_patch is None:
            dt_patch = self.test_data['dt_patch']
        if n_patch is None:
            n_patch = self.test_data['n_patch']
        if index is None:
            index = self.test_data['index']
        if env_ids is None:
            env_ids = self.test_data['env_id']

        # Leave data in list-of-sessions format
        assert len(t_motor_patch) == len(dt_patch) == len(n_patch) == len(index) == len(env_ids)
        if (np.sum(np.array(self.unravel(n_patch)) == 0) == 0) and (len(n_patch) > 0):
            warnings.warn('Patch numbers must be zero-indexed.', category=SyntaxWarning)
        
        # Determine range of tau values for likelihood width calculation.
        if tau_range is None:
            tau_range = self.DEFAULT_TAU_RANGE
            warnings.warn('Range of tau values not specified. Using default values: '
                          '(0.1,...,1000.0, inf).',
                          category=RuntimeWarning)

        # Check filter function if specified.
        if filt is not None and not callable(filt):
            raise SyntaxError('filt must be a callable function.')

        # Iterate over sessions
        self._debug_pred = {'lam': [], 'll': [], 'lhw': []} # DEBUG
        t_hat = []
        t_bin = np.linspace(bin_size, T_max, num=int(T_max/bin_size))
        iters = zip(t_motor_patch, dt_patch, n_patch, index, env_ids)
        for i, (t_m, t_p, n_p, index, env_id) in enumerate(iters):
            # Check environment
            if env_id not in self._active_envs:
                raise RuntimeError('Environments for prediction must be in active'
                                   ' list. Use set_data() to add environment {}.'
                                   .format(env_id))
            
            # Get environment parameters
            self.set_environment(env_id)
            L = int(self.REWARD_VOLUME/self._V0)

            # Get associated priors.
            p_lam0 = self.params['p_lam0'][env_id]
            p_tau = self.params['p_tau'][env_id]

            # Iterate over patches within session
            i_p = np.nonzero(index)[0]
            t_hat.append(np.ones([len(i_p)])*np.nan)
            self._debug_pred['lam'].append(np.ones([len(i_p), len(t_bin)])*np.nan) # DEBUG
            self._debug_pred['ll'].append(np.ones([len(i_p), len(t_bin)])*np.nan) # DEBUG
            self._debug_pred['lhw'].append(np.ones([len(i_p), len(t_bin)])*np.nan) # DEBUG
            for j, m in enumerate(i_p):
                # Get indices corresponding to patches to analyze
                idx = self._get_patch_indices(n_p, m, error=False)
                if idx is None:
                    continue

                # Add reward times if needed.
                t_k = [t_m[k] for k in idx] 
                T = t_p[idx]
                if t_p[m] < T_max:
                    t_start = 0.0 if (len(t_m[m]) == 0) else t_m[m][-1]
                    L0 = self.estimate_unobserved_events(L, t_start, t_p[m]-t_start)
                    t_k[-1] = np.append(t_k[-1], 
                                        self.add_expected_reward_times(L, t_p[m], T_max, N0=L0))
                    T[-1] = T_max
                
                # Iterate over each time bin.
                if filt is None: # without filter, can stop when criteria met
                    t_iters = t_bin
                else: # with filter, must retrieve estimates at all time points first
                    t_iters = [t_bin]
                for k, t in enumerate(t_iters):
                    t_b = np.insert(np.array(t), 0, 0.0)
                    crit = []
                    with np.errstate(divide='raise'):
                        # Compute MLE for Poisson parameters.
                        lam_MLE, tau_MLE = helper.estimate_parameters(t_k, 
                                                                      t_b,
                                                                      L=L,
                                                                      T=T,
                                                                      model='hidden', 
                                                                      history='consecutive',
                                                                      p_lam0=p_lam0,
                                                                      p_tau=p_tau)

                        # Check criterion for Poisson rate.
                        if filt is not None:
                            lam_MLE = filt(lam_MLE)
                        crit.append(lam_MLE <= self.params['lam_thresh'][env_id])
                        
                        # Check criterion for log-likelihood.
                        if self.params['ll_thresh'][env_id] is not None:
                            # Compute log-likelihood for MLE.
                            ll = helper.estimate_log_likelihood(t_k,
                                                                t_b,
                                                                L=L,
                                                                T=T,
                                                                lam=lam_MLE,
                                                                tau=tau_MLE,
                                                                model='hidden', 
                                                                history='consecutive',
                                                                p_lam0=p_lam0,
                                                                p_tau=p_tau,
                                                                epsilon=0.1)
                            ll /= np.sum(T[:-1]) + t # normalize LL by total time of model

                            # Check if criterion met.
                            if filt is not None:
                                ll = filt(ll)
                            crit.append(ll >= self.params['ll_thresh'][env_id])
                        
                        else:
                            crit.append(np.ones([len(t_b)-1], dtype=bool))
                            
                        # Check criterion for log-likelihood width.
                        if self.params['lhw_thresh'][env_id] is not None:
                            # Compute log-likelihood across range of values.
                            lam_range = helper.calculate_parameters(t_k,
                                                                    t_b,
                                                                    L=L,
                                                                    T=T,
                                                                    lam=None,
                                                                    tau=tau_range,
                                                                    model='hidden', 
                                                                    history='consecutive',
                                                                    p_lam0=p_lam0,
                                                                    p_tau=p_tau)
                            #lam_range = lam_range[:, 0] # remove time axis
                            lh = np.ones([len(tau_range), len(t_b)-1])*np.nan
                            for l in range(len(tau_range)):
                                bc = np.ones([len(t_b)-1]) # array for broadcasting
                                lh[l] = helper.estimate_log_likelihood(t_k,
                                                                    t_b,
                                                                    L=L,
                                                                    T=T,
                                                                    lam=lam_range[l]*bc,
                                                                    tau=tau_range[l]*bc,
                                                                    model='hidden', 
                                                                    history='consecutive',
                                                                    p_lam0=p_lam0,
                                                                    p_tau=p_tau,
                                                                    epsilon=0.1)
                            
                            # Estimate peak widths at each time point.
                            # (Unfortunately, this is a difficult method to vectorize.)
                            lhw = np.ones([len(t_b)-1])
                            for l, t in enumerate(t_b[1:]):
                                lhw[l] = self.FWHM(lam_range[:, l], np.exp(lh[:, l]))
                            assert np.sum(lhw < 0.0) == 0

                            # Check if criterion met.
                            if filt is not None:
                                lhw = filt(lhw)
                            crit.append(lhw >= self.params['lhw_thresh'][env_id])

                        else:
                            crit.append(np.ones([len(t_b)-1], dtype=bool))
                    
                    # Determine if criteria for patch-leaving decision are met.
                    # Note that we do not need to scale lam_MLE by L because 
                    # lam_opt is already scaled by L in patch_model.lam.
                    if filt is None:
                        if all(crit):
                            t_hat[-1][j] = t
                            break
                    else:
                        crit = np.logical_and.reduce(tuple(crit), axis=0)
                        if np.sum(crit) > 0:
                            t_hat[-1][j] = t_bin[np.nonzero(crit)[0][0]]
        
        return self.unravel(t_hat)

    #@classmethod
    def smooth_likelihood(self, x, y, sigma=0.25):
        if abs(x[-1] - x[0]) > 0.0:
            # Create smoothing function for estimating likelihood width.
            xi = np.linspace(x[0], x[-1], num=1000)
            yi = np.interp(xi, x, y)
            ys = ephys.smooth_spike_counts(counts=yi,
                                           dt_bin=xi[1]-xi[0],
                                           kernel_type='Gaussian',
                                           sigma=sigma)
            return xi, ys
        else:
            return x, y      

    #@classmethod
    def FWHM(self, x, y):
        # Place data in sorted order. Rather than taking absolute value of
        # difference at end, this allows a better check of proper calculation.
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        # Smooth likelihood to create better behaved function for simple algorithm.
        # Adjust the width of the Gaussian kernel based on the domain, which can
        # vary extensively.
        x, y = self.smooth_likelihood(x, y, sigma=0.02*(x[-1]-x[0]))

        # Find maximum and check expected behavior. Note that this simple
        # algorithm can only handle a single peak.
        idx_max = np.argmax(y)
        if not (np.diff(y[:idx_max]) >= 0.0).all() and (np.diff(y[idx_max:]) <= 0.0).all():
            self._debug_fwhm = (x, y)
            warnings.warn('Likelihood function may have multiple peaks.',
                          category=RuntimeWarning)
        
        # Find x-values corresponding to function values at half-maximum.
        idx = np.argwhere(y < y.max()/2).squeeze()
        if np.sum(idx < idx_max) > 0:
            x1 = x[idx[idx < idx_max][-1]]
        else:
            x1 = None
        if np.sum(idx > idx_max) > 0:
            x2 = x[idx[idx > idx_max][0]]
        else:
            x2 = None
            
        # Return width.
        if x1 is None and x2 is None:
            return np.inf
        elif x1 is None:
            return 2*(x2 - x[idx_max])
        elif x2 is None:
            return 2*(x[idx_max] - x1)
        else:
            return x2 - x1


class FittedBayesianModel(BayesianModel):

    PARAM_NAMES = ['lam_thresh', 'll_thresh', 'lhw_thresh'] # params to fit

    def __init__(self, N):
        super().__init__(N)
        self.name = 'bayes_fit'

        # Set fitted params
        self._param_keys += ['tau_leave']
        self._param_keys += ['logerr']

        # Placeholders for intermediate data
        self._cache = {}

    @property
    def cache(self):
        try:
            return self._cache[self._active_key]
        except KeyError:
            print('Model not yet fit to current dataset.')
            return {}

    def _fit(self,
             data,
             param_names,
             percentile=0.1,
             center=None, 
             constrain=False,
             bin_size=0.5,
             T_max=60.0,
             ftol=0.001,
             max_iters=50,
             tau_range=None,
             filt=None):
        # Check params
        if not isinstance(param_names, list):
            param_names = [param_names]
        if any([name not in self.PARAM_NAMES for name in param_names]):
            raise ValueError('Parameter name not recognized.')

        # Check environment
        if (not constrain) and (len(self._active_envs) > 1):
            warnings.warn('Unconstrained fitting with multiple environments'
                          + ' may result in suboptimal fitting.',
                          category=SyntaxWarning)

        # Leave data in list-of-sessions format
        assert len(data['t_motor_patch']) == len(data['dt_patch']) == len(data['n_patch']) \
               == len(data['index']) == len(data['env_id'])
        if (np.sum(np.array(self.unravel(data['n_patch'])) == 0) == 0) and (len(data['n_patch']) > 0):
            warnings.warn('Patch numbers must be zero-indexed.', category=SyntaxWarning)
        
        # Fit parameters based on experimental values at patch leaving.
        names = [name for name in self.PARAM_NAMES if name not in param_names]
        exp_data, exp_params = self._fit_exp(data, 
                                             names, 
                                             percentile=percentile,
                                             center=center,
                                             constrain=constrain,
                                             tau_range=tau_range)
        
        # Fit parameters based on minimization of prediction error.
        env_ids = np.unique(self.unravel(data['env_id']))
        param_inits = {name: [exp_params[name][env_id] for env_id in env_ids]
                       for name in self.PARAM_NAMES}
        param_inits['lambda0'] = [self.env_params[env_id]['lambda0'] 
                                  for env_id in env_ids]
        self._fit_num(data, 
                      param_names, 
                      param_inits,
                      constrain=constrain,
                      bin_size=bin_size,
                      T_max=T_max,
                      ftol=ftol,
                      max_iters=max_iters,
                      tau_range=tau_range,
                      filt=filt)
                       
        # Rescale Poisson rate based on initial rate
        for env_id in self._params[self._active_key]['lam_thresh'].keys():
            self._params[self._active_key]['lam_thresh'][env_id] *= self.env_params[env_id]['lambda0']
            self._params[self._active_key]['lhw_thresh'][env_id] *= self.env_params[env_id]['lambda0']

        return exp_data

    def _fit_exp(self, 
                 data,
                 param_names,
                 percentile=0.1,
                 center=None, 
                 constrain=False,
                 tau_range=None):
        """
        Compute distributions of estimated Poisson parameters at leaving time,
        including log-likelihood.
        """
        
        # NOTE: Because lambda0 affects the threshold value for leaving by scaling
        # the Poisson rate, we do have multiple environments (different V0) that
        # should theoretically have the same (scaled) parameter lam_thresh. So constrain
        # does apply to these Bayesian models, but by constraining to values of V0, not
        # tau or track.

        # Create placeholders
        env_ids = np.unique(self.unravel(data['env_id']))
        param_keys = self.PARAM_NAMES + ['tau_leave', 'dt_est'] # ignore logerr
        params = {name: {} for name in param_keys}

        # Determine range of tau values for likelihood width calculation.
        if tau_range is None:
            tau_range = self.DEFAULT_TAU_RANGE
            warnings.warn('Range of tau values not specified. Using default values: '
                          '(0.1,...,1000.0, inf).',
                          category=RuntimeWarning)

        # Iterate over sessions
        iters = zip(data['t_motor_patch'], data['dt_patch'], data['n_patch'], 
                    data['dt_interpatch'], data['index'], data['env_id'])
        for i, (t_m, t_p, n_p, t_ip, index, env_id) in enumerate(iters):
            # Check environment
            if env_id not in self._active_envs:
                raise RuntimeError('Environments must be in active'
                                   ' list. Use set_data() to add environment {}.'
                                   .format(env_id))
            
            # Get environment parameters
            self.set_environment(env_id)
            r_0 = self._V0*self._lambda0
            tau = self._tau
            track = self._track
            L = int(self.REWARD_VOLUME/self._V0)

            # Add travel time data
            if constrain:
                key = track
            else:
                key = 'all'
            if key in params['dt_est'].keys():
                params['dt_est'][key].append(t_ip[index])
            else:
                params['dt_est'][key] = [t_ip[index]]

            # Iterate over patches within session
            i_p = np.nonzero(index)[0] # patches to index
            lt = np.ones([len(i_p)])*np.nan # lambda threshold
            tl = np.ones([len(i_p)])*np.nan # tau at leaving
            llt = np.ones([len(i_p)])*np.nan # log-likelihood threshold
            lhw = np.ones([len(i_p)])*np.nan # likelihood width
            for j, m in enumerate(i_p):
                # Get indices corresponding to patches to analyze.
                idx = self._get_patch_indices(n_p, m, error=False)
                if idx is None:
                    continue

                # Get reward and patch residence times.
                t_k = [t_m[k] for k in idx] 
                T = t_p[idx]
                
                # Find MLE for Poisson rate and corresponding likelihood
                # at patch leaving.
                with np.errstate(divide='raise'):
                    # Compute MLE for Poisson parameters.
                    lam_MLE, tau_MLE = helper.estimate_parameters(t_k, 
                                                                  np.array([0.0, t_p[m]]),
                                                                  L=L,
                                                                  T=T,
                                                                  model='hidden', 
                                                                  history='consecutive')
                    
                    # Compute log-likelihood for MLE.
                    ll = helper.estimate_log_likelihood(t_k,
                                                        np.array([0.0, t_p[m]]),
                                                        L=L,
                                                        T=T,
                                                        lam=lam_MLE,
                                                        tau=tau_MLE,
                                                        model='hidden', 
                                                        history='consecutive', 
                                                        epsilon=0.1)
                    ll /= np.sum(T[:-1]) + t_p[m] # normalize LL by total time of model

                    # Compute log-likelihood across range of values.
                    lam_range = helper.calculate_parameters(t_k,
                                                            np.array([0.0, t_p[m]]),
                                                            L=L,
                                                            T=T,
                                                            lam=None,
                                                            tau=tau_range,
                                                            model='hidden', 
                                                            history='consecutive')
                    lam_range = lam_range[:, 0] # remove time axis
                    lh = np.ones([len(tau_range)])*np.nan
                    for k in range(len(tau_range)):
                        lh[k] = helper.estimate_log_likelihood(t_k,
                                                               np.array([0.0, t_p[m]]),
                                                               L=L,
                                                               T=T,
                                                               lam=np.atleast_1d(lam_range[k]),
                                                               tau=np.atleast_1d(tau_range[k]),
                                                               model='hidden', 
                                                               history='consecutive', 
                                                               epsilon=0.1)
                
                # Save results
                lt[j] = lam_MLE/self._lambda0 # normalize Poisson rate
                tl[j] = tau_MLE
                llt[j] = ll
                lhw[j] = self.FWHM(lam_range, np.exp(lh))/self._lambda0 # normalize
            
            # Add estimated parameter data
            for k, v in zip(['lam_thresh', 'tau_leave', 'll_thresh', 'lhw_thresh'], [lt, tl, llt, lhw]):
                if constrain and k in ['lam_thresh', 'tau_leave']:
                    key = (tau, track)
                elif constrain and k in ['ll_thresh', 'lhw_thresh']:
                    key = self._V0
                else:
                    key = 'all'
                if key in params[k].keys():
                    params[k][key].append(v)
                else:
                    params[k][key] = [v]
            
        # Determine model parameter values for each data subset.
        raw = copy.deepcopy(params) # copy all values to return
        for key, vals in params.items():
            # Determine metric
            if isinstance(percentile, dict):
                p = percentile.get(key, None)
            else:
                p = percentile
            if isinstance(center, dict):
                c = center.get(key, None)
            else:
                c = center
            method = self._get_method(p, c)

            # Apply metric to each data subset.
            for k, v in vals.items():
                v = np.hstack(v)
                v = v[np.isfinite(v)] # filter out negative inf for log-likelihood
                params[key][k] = method(v)

        # Remap values to environment IDs. For unconstrained fit,
        # single value will be mapped to all environments.
        temp = {name: {} for name in param_keys}
        for env_id in env_ids:
            tau = self.env_params[env_id]['tau']
            track = self.env_params[env_id]['track']
            V0 = self.env_params[env_id]['V0']
            for key, vals in params.items():
                for k, v in vals.items():
                    if isinstance(k, tuple):
                        match = (k == (tau, track))
                    elif constrain and key not in ['ll_thresh', 'lhw_thresh']:
                        match = (k == track)
                    elif constrain and key in ['ll_thresh', 'lhw_thresh']:
                        match = (k == V0)
                    else:
                        match = (k == 'all')
                    
                    if match:
                        temp[key][env_id] = v
                        break
        params = temp
        
        # Save fitted parameters. Note that lam_thresh is normalized by the
        # initial Poisson rate and will be rescaled at the end of fitting.
        for name in param_names + ['tau_leave', 'dt_est']:
            self._params[self._active_key][name] = params[name]
        
        return raw, copy.deepcopy(params) # copy normalized params for numerical fit

    def _fit_num(self,
                 data,
                 param_names,
                 param_inits,
                 constrain=False,
                 bin_size=0.5,
                 T_max=60.0,
                 ftol=0.001,
                 max_iters=50,
                 tau_range=None,
                 filt=None):
        """
        Compute threshold parameters by minimizing error function for predictions
        of residence times for training data.
        """
        # NOTE: We need to pass the initial values for lam_thresh as normalized, and
        # then rescale in the error function.
        # START HERE: Need to work on incorporating constrained fits to numerical optimization
        # since using the mean lam_thresh and ll_thresh doesn't work well.
        # Should lam_thresh be constrained to same values of (tau, track) and ll_thresh
        # to same values of V0?

        # NOTE: Because lambda0 affects the threshold value for leaving by scaling
        # the Poisson rate, we do have multiple environments (different V0) that
        # should theoretically have the same (scaled) parameter lam_thresh. So constrain
        # does apply to these Bayesian models, but by constraining to values of V0, not
        # tau or track.

        # Create placeholders
        env_ids = np.unique(self.unravel(data['env_id']))
        lam_all = {env_id: [] for env_id in env_ids}
        ll_all = {env_id: [] for env_id in env_ids}
        lhw_all = {env_id: [] for env_id in env_ids}
        Y = {env_id: [] for env_id in env_ids}
        t_bin = np.linspace(0.0, T_max, num=int(T_max/bin_size)+1)

        # Determine range of tau values for likelihood width calculation.
        if tau_range is None:
            tau_range = self.DEFAULT_TAU_RANGE
            warnings.warn('Range of tau values not specified. Using default values: '
                          '(0.1,...,1000.0, inf).',
                          category=RuntimeWarning)

        # Check filter function if specified.
        if filt is None:
            filt = lambda x: x
        elif not callable(filt):
            raise SyntaxError('filt must be a callable function.')
        
        # Iterate over sessions
        if self._cache.get(self._active_key, None) is None:
            iters = zip(data['t_motor_patch'], data['dt_patch'], data['n_patch'], 
                        data['dt_interpatch'], data['index'], data['env_id'])
            for i, (t_m, t_p, n_p, t_ip, index, env_id) in enumerate(iters):
                # Check environment
                if env_id not in env_ids:
                    raise RuntimeError('Environments must be in active'
                                    ' list. Use set_data() to add environment {}.'
                                    .format(env_id))
                
                # Get environment parameters
                self.set_environment(env_id)
                r_0 = self._V0*self._lambda0
                tau = self._tau
                track = self._track
                L = int(self.REWARD_VOLUME/self._V0)

                # Iterate over patches within session
                i_p = np.nonzero(index)[0] # patches to index
                lam_i = np.ones([len(i_p), len(t_bin)-1])*np.nan # estimated lambda at each time point
                ll_i = np.ones([len(i_p), len(t_bin)-1])*np.nan # log-likelihood at each time point
                lhw_i = np.ones([len(i_p), len(t_bin)-1])*np.nan # likelihood width at each time point
                for j, m in enumerate(i_p):
                    # Get indices corresponding to patches to analyze.
                    idx = self._get_patch_indices(n_p, m, error=False)
                    if idx is None:
                        continue

                    # Get reward and patch residence times.
                    t_k = [t_m[k] for k in idx] 
                    T = t_p[idx]
                    if t_p[m] < T_max:
                        t_start = 0.0 if (len(t_m[m]) == 0) else t_m[m][-1]
                        L0 = self.estimate_unobserved_events(L, t_start, t_p[m]-t_start)
                        t_k[-1] = np.append(t_k[-1], 
                                            self.add_expected_reward_times(L, t_p[m], T_max, N0=L0))
                        T[-1] = T_max

                    # Find MLE for Poisson rate and corresponding likelihood for
                    # all time points in t_bin, which will be used for error minimization.
                    with np.errstate(divide='raise'):
                        # Compute MLE for Poisson parameters.
                        lam_MLE, tau_MLE = helper.estimate_parameters(t_k, 
                                                                      t_bin,
                                                                      L=L,
                                                                      T=T,
                                                                      model='hidden', 
                                                                      history='consecutive')
                        
                        # Compute log-likelihood for MLE.
                        ll = helper.estimate_log_likelihood(t_k,
                                                            t_bin,
                                                            L=L,
                                                            T=T,
                                                            lam=lam_MLE,
                                                            tau=tau_MLE,
                                                            model='hidden', 
                                                            history='consecutive', 
                                                            epsilon=0.1)
                        ll /= np.sum(T[:-1]) + t_bin[1:] # normalize LL by total time of model

                        # Compute log-likelihood across range of values.
                        lam_range = helper.calculate_parameters(t_k,
                                                                t_bin,
                                                                L=L,
                                                                T=T,
                                                                lam=None,
                                                                tau=tau_range,
                                                                model='hidden', 
                                                                history='consecutive')
                        lh = np.ones([len(tau_range), len(t_bin)-1])*np.nan
                        for k in range(len(tau_range)):
                            lh[k, :] = helper.estimate_log_likelihood(t_k,
                                                                      t_bin,
                                                                      L=L,
                                                                      T=T,
                                                                      lam=lam_range[k, :],
                                                                      tau=tau_range[k]*np.ones([len(t_bin)-1]),
                                                                      model='hidden', 
                                                                      history='consecutive', 
                                                                      epsilon=0.1)
                        
                        # Estimate peak widths at each time point.
                        # (Unfortunately, this is a difficult method to vectorize.)
                        lhw = np.ones([len(t_bin)-1])
                        for k, t in enumerate(t_bin[1:]):
                            lhw[k] = self.FWHM(lam_range[:, k], np.exp(lh[:, k]))
                        if np.sum(lhw < 0.0) > 0:
                            self._debug = {'lam_range': lam_range,
                                           'lh': lh,
                                           'lhw': lhw,
                                           'lam_MLE': lam_MLE,
                                           'tau_MLE': tau_MLE,
                                           'll': ll,
                                           't_k': t_k,
                                           't_bin': t_bin,
                                           'L': L,
                                           'T': T}
                            raise ValueError('Negative lhw.')

                    # Save results
                    lam_i[j, :] = filt(lam_MLE)
                    ll_i[j, :] = filt(ll)
                    lhw_i[j, :] = filt(lhw)

                # Add discretized estimated values for optimization function.
                lam_all[env_id].append(lam_i)
                ll_all[env_id].append(ll_i)
                lhw_all[env_id].append(lhw_i)
                Y[env_id].append(t_p[i_p]) # residence times (true value for error function)
            
            # Cache computationally expensive values.
            self._cache[self._active_key] = {'lam': lam_all, 
                                             'll': ll_all, 
                                             'lhw': lhw_all,
                                             'Y': Y}
        
        else:
            # Retrieve from previous computation.
            lam_all, ll_all, lhw_all, Y = \
                self.cache['lam'], self.cache['ll'], self.cache['lhw'], self.cache['Y']
        
        # Format inputs to optimization function as list of values for each
        # environment, with env_id order preserved.
        lam_all = [np.vstack(lam_all[env_id]) for env_id in env_ids]
        ll_all = [np.vstack(ll_all[env_id]) for env_id in env_ids]
        lhw_all = [np.vstack(lhw_all[env_id]) for env_id in env_ids]
        Y = [np.hstack(Y[env_id]) for env_id in env_ids]
        x_init, params = self._build_opt_inputs(param_names,
                                                param_inits,
                                                constrain=constrain)

        # Fit parameters numerically to minimize prediction error.
        error = 0.0
        self._debug = {'lam_all': lam_all, 'll_all': ll_all, 'lhw_all': lhw_all, 'Y': Y, 
                       'params': params, 'x_init': x_init , 't_bin': t_bin}
        for i in range(max_iters):
            with np.errstate(invalid='raise'):
                options = {'fatol': 1e-4, 
                           'xatol': 1e-4, 
                           'adaptive': True,
                           'initial_simplex': self._build_initial_simplex(x_init, params)}
                res = opt.minimize(self.bayesian_error, 
                                   x_init, 
                                   args=(Y, params, lam_all, ll_all, lhw_all, t_bin),
                                   method='Nelder-Mead',
                                   options=options)
            if abs(error - res.fun) <= ftol:
                break
            elif i == max_iters - 1:
                warnings.warn('Maximum iterations exceeded.', category=RuntimeWarning)
            else:
                x_init = res.x
                error = res.fun

        # Save optimization results
        if res.success:
            for var in param_names:
                vals = np.atleast_1d(res.x[params[var]['index']])[params[var]['ids']]
                self._params[self._active_key][var] = \
                    {env_id: v for env_id, v in zip(env_ids, vals)}
            self._params[self._active_key]['logerr'] = res.fun
        else:
            for var in param_names:
                self._params[self._active_key][var] = \
                    {env_id: np.nan for env_id in env_ids}
            self._params[self._active_key]['logerr'] = np.nan

    def _get_method(self, percentile, center):
        # Get metric for determining parameter value from distributions.
        # Note that percentile, if defined, gets precedence.
        if percentile is not None:
            assert 0.0 <= percentile <= 1.0
            method = lambda x: np.sort(x)[int(percentile*(len(x)-1))]
        elif center == 'mean':
            method = np.mean
        elif center == 'median':
            method = np.median
        else:
            raise ValueError('Unknown metric \'{}\'.'.format(center))
    
        return method

    @staticmethod
    def bayesian_error(x, Y, params, lam_MLE, ll_MLE, lhw, t_bin):
        # Set parameter values
        loc = []
        for var in ['lam_thresh', 'll_thresh', 'lhw_thresh', 'lambda0']:
            idx = params[var].get('index', None)
            if idx is None:
                # If index not provided, pass value of parameter in each condition.
                loc.append(np.atleast_1d(params[var]['value']))
            else:
                # Use index to index x for unique values, and then use ids to expand
                # vector in order to assign value for each condition.
                loc.append(np.atleast_1d(x[params[var]['index']][params[var]['ids']]))

        # Get threshold values from optimization vector.        
        lam_thresh, ll_thresh, lhw_thresh, lambda0 = loc
        assert len(Y) == len(lam_thresh) == len(ll_thresh) == len(lhw_thresh) \
               == len(lambda0) == len(lam_MLE) == len(ll_MLE) == len(lhw)

        # Estimated lambda and log-likelihood should be provided as list
        # of arrays, where each array represents aggregated patch data for
        # an environment.
        y_hat = []
        t_bin = t_bin[1:]
        iters = zip(lam_MLE, ll_MLE, lhw, lam_thresh, ll_thresh, lhw_thresh, lambda0)
        for lam_est, ll_est, lhw_est, lam_th, ll_th, lhw_th, lam0 in iters:
            # Ensure arrays are 2D.
            lam_est, ll_est, lhw_est = np.atleast_2d(lam_est, ll_est, lhw_est)
            assert np.sum(lhw_est < 0.0) == 0

            # Find first time indices that satisfy criteria.
            crit = (lam_est <= lam_th*lam0, 
                    ll_est >= ll_th, 
                    lhw_est <= lhw_th*lam0)
            crit = np.logical_and.reduce(crit, axis=0)
            idx = np.argwhere(crit)
            N, index = np.unique(idx[:, 0], return_index=True)
            idx = idx[index, 1]        

            # Save predictions.
            pred = np.ones([lam_est.shape[0]])*np.nan
            pred[N] = t_bin[idx]
            pred[np.isnan(pred)] = t_bin[-1] # TODO: is this the right penalty?
            y_hat.append(pred)
        
        # Compute error among condition sets
        error = []
        for i in range(len(Y)):
            error.append((Y[i] - y_hat[i])**2)
        
        # Return logsum of squared error (more numerically stable)
        return np.log(np.sum(np.hstack(error)))

    def _build_initial_simplex(self, x0, params):
        N = len(x0)
        sim = np.ones([N+1, N])*x0
        for name, vals in params.items():
            if 'index' not in vals.keys():
                continue
            elif name == 'lam_thresh':
                scale = 1.05
            elif name == 'll_thresh':
                scale = 1.5
            elif name == 'lhw_thresh':
                scale = 1.5
            slc = vals['index']
            sim[0, slc] /= scale
            size = (slc.stop - slc.start)
            sim[1:][slc, slc][np.diag_indices(size)] *= scale
        return sim