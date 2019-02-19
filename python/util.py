import numpy as np
import os
import re

### General utility functions ###
def _check_list(names):
    """
    Changes variable to list if not already an instance of one.
    """
    if not isinstance(names, list):
        names = [names]
    return names

### File handling ###
def find_files(path, files):
    """Recursive algorithm for finding files"""
    # If path is file, then append to list
    if os.path.isfile(path):
        files.append(path)
        return
    
    # Otherwise, iterate through all children (files/subdirectories)
    for f in os.listdir(path):
        _ = find_files(os.path.join(path, f), files)    
        
    return files

def find_data(mouse_id, files, ext='.mat', exclude_strs=[]):
    """Find data files and associated session numbers for given mouse ID."""
    # Filter files by extension, mouse ID
    filelist = []
    exclude_strs = _check_list(exclude_strs)
    for f in files:
        exclude = False
        for exclude_str in exclude_strs:
            if exclude_str in f.lower():
                exclude = True
        if (f.lower().endswith(ext)
            and mouse_id in f.lower()
            and not exclude):
            filelist.append(f)

    # Find training days of all files
    training_days = np.zeros(len(filelist), dtype=np.int16)
    for i, filename in enumerate(filelist):
        match = re.search('_d[0-9]+[a-z]*_', filename, re.IGNORECASE)
        if match is not None:
            day = match.group()[2:-1]
            match = re.search('[a-z]+', day, re.IGNORECASE)
            if match is not None:
                day = day[:match.span()[0]]
            training_days[i] = int(day)
        else:
            training_days[i] = -1

    # Sort filenames and save associated training days
    keep_idx = (training_days >= 0)
    training_days = training_days[keep_idx]
    filelist = [f for i, f in enumerate(filelist) if keep_idx[i]]
    sort_idx = np.argsort(training_days)
    filelist = [filelist[i] for i in sort_idx]
    training_days = np.sort(training_days)

    return filelist, training_days

### Array handling ###
def in_interval(t, t1, t2, query='event', include_border=True):
    # TODO: complete boolean for include_border
    gt_t1 = (t[np.newaxis, :] >= t1[:, np.newaxis])
    lt_t2 = (t[np.newaxis, :] <= t2[:, np.newaxis])
    bt_t1_t2 = np.logical_and(gt_t1, lt_t2)

    if query == 'event':
        return np.sum(bt_t1_t2.astype(np.int32), axis=0)
    elif query == 'interval':
        return np.sum(bt_t1_t2.astype(np.int32), axis=1)
    elif query == 'array':
        return bt_t1_t2
    else:
        raise SyntaxError('Unknown query \'%s\'.' % query)

def remove_outliers(a, thresh=2.0):
    mean = np.mean(a)
    std = np.std(a)
    return a[np.logical_and(a > mean-thresh*std, a < mean+thresh*std)]

def flatten_list(a, ids=None):
    """Converts list of arrays into single 1D array."""
    # Determine total number of elements in list
    n = 0
    for a_i in a:
        if isinstance(a_i, np.ndarray):
            n += a_i.size
        elif isinstance(a_i, list):
            n += len(a_i)
        else:
            n += 1
        
    # Build flattened array
    a_flat = np.zeros(n)
    ids_flat = np.zeros(n)
    j = 0
    
    if ids is not None:
        for i, [a_i, idx] in enumerate(zip(a, ids)):
            if isinstance(a_i, np.ndarray):
                a_flat[j:j+a_i.size] = a_i.flatten()
                if isinstance(idx, list) or isinstance(idx, np.ndarray):
                    ids_flat[j:j+a_i.size] = np.asarray(idx)
                else:
                    ids_flat[j:j+a_i.size] = idx * np.ones(a_i.size)
                j += a_i.size
            elif isinstance(a_i, list):
                a_flat[j:j+len(a_i)] = np.asarray(a_i)
                if isinstance(idx, list) or isinstance(idx, np.ndarray):
                    ids_flat[j:j+len(a_i)] = np.asarray(idx)
                else:
                    ids_flat[j:j+len(a_i)] = idx * np.ones(len(a_i))
                j += len(a_i)
            else:
                a_flat[j] = np.asarray(a_i)
                if isinstance(idx, list) or isinstance(idx, np.ndarray):
                    ids_flat[j] = np.asarray(idx)
                else:
                    ids_flat[j] = idx
                j += 1
                
        return a_flat, ids_flat
    
    else:
        for i, a_i in enumerate(a):
            if isinstance(a_i, np.ndarray):
                a_flat[j:j+a_i.size] = a_i.flatten()
                j += a_i.size
            elif isinstance(a_i, list):
                a_flat[j:j+len(a_i)] = np.asarray(a_i)
                j += len(a_i)
            else:
                a_flat[j] = np.asarray(a_i)
                j += 1
    
        return a_flat
    
def get_patch_statistics(stats, 
                         ids, 
                         *args, 
                         method='mean', 
                         ignore_nan=True, 
                         return_all=False):
    """
    Returns populations statistics for given input over sessions (and animals).
    
    Args:
    - stats: Either list of the statistic for each session, or a dictionary of such lists
             over multiple animals. Note that the data type must be the same for each
             session (e.g. numpy array, list, int).
    - ids: Either numpy array of identifiers corresponding to the sessions, or a 
           dictionary of such arrays over multiple animals.
    - *args: Other identifiers in the same format as ids, but which will not be used to
             cluster data points.
    - method: The method to use on the data points for each id (e.g. mean, median).
    - ignore_nan: If True, ignore NaN values in array.
    - return_all: If True, also return raw data points and corresponding IDs. Otherwise,
                  return only specified metric.
    
    Returns:
    - stats_all: 1D-array or list of all statistics over all patches, sessions, and animals
    - ids_all: Corresponding identifiers for each entry in stats_all 
    """
    n_args = len(args)
    
    # Get flattened array or list of data
    if isinstance(stats, dict):
        # Reformat args
        if n_args > 0:
            args_ = {}
            for k in args[0].keys():
                args_[k] = []
            for arg in args:
                for k, v in arg.items():
                    args_[k].append(v)
            args = args_
        else:
            args = {}
            for k in stats.keys():
                args[k] = []
           
        stats_all = []
        ids_all = []
        args_all = []
        for _ in range(n_args):
            args_all.append([])
        for [_, stat], [_, idx], [_, args_i] in zip(stats.items(), ids.items(), args.items()):
            stats_all_, ids_all_  = flatten_list(stat, ids=idx)
            stats_all.append(stats_all_)
            ids_all.append(ids_all_)
            for i in range(n_args):
                _, args_all_i = flatten_list(stat, ids=args_i[i])
                args_all[i].append(args_all_i)
            
        args_all_ = []
        for _ in range(n_args):
            args_all_.append([])
        for i in range(n_args):
            _, args_all_i = flatten_list(stats_all, ids=args_all[i])
            args_all_[i] = args_all_i
        args_all = args_all_
        stats_all, ids_all = flatten_list(stats_all, ids=ids_all)
        
    
    else:
        stats_all, ids_all = flatten_list(stats, ids=ids)
        args_all = []
        for i in range(n_args):
            _, args_i = flatten_list(stats_all, ids=args[i])
            args_all.append(args_i)
    
    # Calculate desired metric for all points with given id
    if method == 'mean':
        f = np.mean
    elif method == 'median':
        f = np.median
    elif method == 'std':
        f = np.std
    elif method == 'sem':
        f = lambda x: np.std(x) / np.sqrt(x.size)
    else:
        raise ValueError('Method %s not supported.' % method)
    
    stats_id = np.zeros(len(np.unique(ids_all)))
    ids_id = np.unique(ids_all)
    args_id = []
    for _ in range(n_args):
        args_id.append([])
    for i, idx in enumerate(ids_id):
        idx_ = (ids_all == idx)
        if ignore_nan:
            idx_[np.isnan(stats_all)] = False
        stats_id[i] = f(stats_all[idx_])
        for i in range(n_args):
            args_id[i].append(args_all[i][idx_])
    
    if n_args > 0:
        if return_all:
            return stats_all, ids_all, args_all
        else:
            return stats_id, ids_id, args_id
    else:
        if return_all:
            return stats_all, ids_all
        else:
            return stats_id, ids_id