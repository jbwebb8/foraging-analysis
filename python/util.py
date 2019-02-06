import numpy as np
import os

### General utility functions ###
def _check_list(names):
    """
    Changes variable to list if not already an instance of one.
    """
    if not isinstance(names, list):
        names = [names]
    return names

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

### Array handling ###
def in_interval(t, t1, t2, query='event'):
    gt_t1 = (t[np.newaxis, :] > t1[:, np.newaxis])
    lt_t2 = (t[np.newaxis, :] < t2[:, np.newaxis])
    bt_t1_t2 = np.logical_and(gt_t1, lt_t2)

    if query == 'event':
        return np.sum(bt_t1_t2.astype(np.int32), axis=0)
    elif query == 'interval':
        return np.sum(bt_t1_t2.astype(np.int32), axis=1)
    elif query == 'array':
        return bt_t1_t2
    else:
        raise SyntaxError('Unknown query \'%s\'.' % query)

def flatten_list(a):
    """Converts list of arrays into single 1D array"""
    # Determine total number of elements in list
    n = 0
    for a_i in a:
        n += a_i.size
        
    # Build flattened array
    a_flat = np.zeros(n)
    j = 0
    for i, a_i in enumerate(a):
        a_flat[j:j+a_i.size] = a_i.flatten()
        j += a_i.size
    
    return a_flat

def remove_outliers(a, thresh=2.0):
    mean = np.mean(a)
    std = np.std(a)
    return a[np.logical_and(a > mean-thresh*std, a < mean+thresh*std)]