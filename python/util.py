# General utility packages
import os
import re
import struct
import time

# Google Drive API
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Numerical
import numpy as np

### General utility functions ###
def _check_list(names):
    """
    Changes variable to list if not already an instance of one.
    """
    if not isinstance(names, list):
        names = [names]
    return names

def recursive_dict_search(dictionary, old_value, new_value):
    """
    Replaces all key-old_value pairs with key-new_value pairs in dictionary.
    
    Args:
    - d: Dictionary to be modified.
    - key: Initial key.
    - old_value: Value to be replaced.
    - new_value: Value to replace old_value.
    Returns:
    - None (dictionary modified in place)
    """
    _recursive_dict_search(dictionary, dictionary, [], old_value, new_value)

def _recursive_dict_search(init_d, d, keys, old_val, new_val):
    # If dictionary, continue to explore all key-value pairs
    if isinstance(d, dict):
        for k, v in zip(d.keys(), d.values()):
            keys.append(k)
            _recursive_dict_search(init_d, v, keys, old_val, new_val)
        if len(keys) > 0: # avoids error at end
            keys.pop()
    
    # If list, then value list iterated through for all items
    elif isinstance(d, list):
        d = _recursive_list_search(d, old_val, new_val)
        t = init_d 
        for key in keys[:-1]:
            t = t[key]
        t[keys[-1]] = d
        keys.pop()
    
    # Otherwise, then replace old value with new value
    else:
        if d == old_val:
            t = init_d 
            for key in keys[:-1]:
                t = t[key]
            t[keys[-1]] = new_val
        keys.pop()

def _recursive_list_search(l, old_val, new_val):
    if isinstance(l, list):
        t = []
        for l_ in l:
            v = _recursive_list_search(l_, old_val, new_val)
            t.append(v)
        return t
    elif isinstance(l, dict):
        _recursive_dict_search(l, l, [], old_val, new_val)
        return l
    else:
        if l == old_val:
            l = new_val
        return l

### File handling ###
class GoogleDriveHandler:
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    def __init__(self, cred_file='credentials.json'):
        # Credentials placeholder
        creds = None

        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
                
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            # Grab credentials from JSON file
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    cred_file, SCOPES)
                creds = flow.run_local_server()
            
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        # Build service
        self.service = build('drive', 'v3', credentials=creds)

    def download(self, *, filename=None, file_id=None, byte_range=None):
        # Get download url
        if file_id is None:
            file_id = self.get_file_id(filename)
        fields = self.service.files().get(fileId=file_id, fields='webContentLink, size').execute()
        download_url = fields['webContentLink']
        size = int(fields['size'])
        
        # Get file content
        if byte_range is None:
            byte_range = [0, size-1]
        header = {'Range': 'bytes=%d-%d' % (byte_range[0], byte_range[1])}
        resp, content = self.service._http.request(download_url, headers=header)

        return content

    def get_file_id(self, filename):
        results = self.service.files().list(q="name contains '%s'" % filename).execute()
        return results['files']


class RHDHeader:

    def __init__(self, *, f=None, filepath=None):
        # Load file if not provided
        if f is None:
            f = open(filepath, 'rb')
            close_file = True
        else:
            close_file = False

        # Intan header
        self.header, = struct.unpack('I', f.read(4))
        
        # Version number
        self.version_major, = struct.unpack('h', f.read(2))
        self.version_minor, = struct.unpack('h', f.read(2))
        
        # Sample rate
        self.f_s = int(struct.unpack('f', f.read(4))[0])

        if close_file:
            f.close()
        else:
            f.seek(0, 0) # reset pointer


class MDAHeader:

    def __init__(self, *, f=None, filepath=None):
        # Load file if not provided
        if f is None:
            f = open(filepath, 'rb')
            close_file = True
        else:
            close_file = False
        
        # Placeholder for header length
        self.n_bytes_header = 0

        # First 32 bits: signed integer indicating data type
        self.dtype_code, = struct.unpack('i', f.read(4))
        self.n_bytes_header += 4

        # Second 32 bits: number of bytes per entry
        self.n_bytes_entry, = struct.unpack('i', f.read(4))
        self.n_bytes_header += 4

        # Third 32 bits: number of dimensions
        self.n_dims, = struct.unpack('i', f.read(4))
        self.n_bytes_header += 4

        # Next 32*n_dims bits: size of each dimension
        self.shape = np.zeros(self.n_dims, dtype=np.int64)
        for i in range(self.n_dims):
            self.shape[i], = struct.unpack('i', f.read(4))
            self.n_bytes_header += 4

        # Determine data type from code
        if self.dtype_code == -3: # float32
            #self.n_bytes_entry = 4
            self.dtype_char = 'f'
            self.dtype_np = np.float32
        elif self.dtype_code == -4: # int16
            #self.n_bytes_entry = 2
            self.dtype_char = 'h'
            self.dtype_np = np.int16
        elif self.dtype_code == -5: # int32
            #self.n_bytes_entry = 4
            self.dtype_char = 'i'
            self.dtype_np = np.int32
        elif self.dtype_code == -6: # uint16
            #self.n_bytes_entry = 2
            self.dtype_char = 'H'
            self.dtype_np = np.uint16
        elif self.dtype_code == -7: # double
            #self.n_bytes_entry = 8
            self.dtype_char = 'd'
            self.dtype_np = np.float64
        elif self.dtype_code == -8: # uint32
            #self.n_bytes_entry = 4
            self.dtype_char = 'I'
            self.dtype_np = np.uint32
        else:
            raise ValueError('Unknown code (%d).' % code)
        
        if close_file:
            f.close()
        else:
            f.seek(0, 0) # reset pointer


class MDAReader:

    def __init__(self, filepath, drive_file=False, service=None):
        """
        Reads MDA files generated by MountainSort software. Assumes data shape is
        [num_channels, num_samples].

        Args:
        - filepath: If drive_file=False, local location of file. If drive_file=True,
                    file_id of file.
        - drive_file: True if providing Google Drive file (and service via API).
        - service: Google Drive API service if using Google Drive file.
        """
        if drive_file:
        self.f = open(filepath, 'rb')
        self.header = MDAHeader(f=self.f)# Shape values
        self.num_channels = self.header.shape[0]
        self.N = self.header.shape[1]

    def read(self, *, 
             ch_start=None, 
             ch_end=None, 
             sample_start=None, 
             sample_end=None,
             io_attempts=5):
        # Default indices (0-indexed)
        if ch_start == None:
            ch_start = 0
        if ch_end == None:
            ch_end = self.num_channels - 1
        if sample_start == None:
            sample_start = 0
        if sample_end == None:
            sample_end = self.N - 1
        
        # Set pointer to first byte of slice
        byte_start = self.header.n_bytes_header + (self.header.n_bytes_entry * self.num_channels * sample_start)
        self.f.seek(byte_start, 0)
        
        # Get specified entries
        # For efficiency, just read all channels, and then slice at end
        num_samples = sample_end - sample_start
        num_pts = self.num_channels * num_samples
        format_str = '%d%s' % (num_pts, self.header.dtype_char)
        b = None # binary placeholder
        for _ in range(io_attempts):
            # Reading from streamed and/or large file can occasionally throw errors.
            # Sometimes, this is resolved by subsequent calls.
            try:
                b = self.f.read(self.header.n_bytes_entry*num_pts)
                break
            except IOError:
                print('IOError: Retrying in one second...')
                time.sleep(1)
                self.f.seek(byte_start, 0) # reset pointer just in case
        
        # Convert to numpy array if successful
        if b is not None:
            X = np.array(struct.unpack(format_str, b))
        else:
            raise IOError('Error reading file.')
        #X = np.fromfile(self.f, dtype=self.header.dtype_np, count=num_pts)

        return X.reshape([self.num_channels, num_samples], order='F')[ch_start:ch_end+1, :]

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
                    print(j)
                    print(idx)
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