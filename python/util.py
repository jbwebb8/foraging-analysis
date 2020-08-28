# General utility packages
import os
import re
import struct
import time
from tempfile import TemporaryFile
import json
import gc

# Google Drive API
try:
    import pickle
    import os.path
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.errors import HttpError
except ModuleNotFoundError as e:
    print(e, 'Some Google Drive API methods may not be available.')

# Numerical
import numpy as np

# Plotting
import matplotlib.pyplot as plt

### General utility functions ###
def _check_list(names):
    """
    Changes variable to list if not already an instance of one.
    """
    if not isinstance(names, list):
        names = [names]
    return names

def load_json(filepath, auto_replace=True):
    """
    Loads JSON file from filename into python dictionary.

    Args:
    - filepath (str): Location of JSON file to load.
    - auto_replace (bool): If True, automically replace common data types
        formatted as strings in JSON file to appropriate Python data types
        (e.g. boolean, None)
    """
    with open(filepath, 'r') as f:
        d = json.loads(f.read())
    
    if auto_replace:
        replace_dict = {'True': True,
                        'true': True,
                        'False': False,
                        'false': False,
                        'None': None,
                        'none': None}
        for v_old, v_new in replace_dict.items():
            recursive_dict_search(d, v_old, v_new)
    
    return d

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

def format_elapsed_time(dt):
    h = dt // 3600
    m = (dt - (3600*h)) // 60
    s = dt - (3600*h) - (60*m)
    return h, m, s

### File handling ###
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class HTTPError(Error):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        print(message)

class GoogleDriveService:
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Base URIs for downloads and uploads
    BASE_DOWNLOAD_URI = 'https://www.googleapis.com/drive/v3/files'
    BASE_UPLOAD_URI = 'https://www.googleapis.com/upload/drive/v3/files'

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
                # Check for credentials file
                if not os.path.exists(cred_file):
                    raise SyntaxError('Credentials file %s does not exist. Follow '
                                      'instructions for downloading file at Google '
                                      'Drive API website, and place in current directory.'
                                      % cred_file)
                flow = InstalledAppFlow.from_client_secrets_file(
                    cred_file, self.SCOPES)
                creds = flow.run_local_server()
            
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        # Build service
        self._service = build('drive', 'v3', credentials=creds)

    def download(self, *, 
                 filename=None, 
                 file_id=None, 
                 byte_range=None,
                 chunk_size=100*1024*1024, # from Google API
                 file_object=None,
                 verbose=True):
        # Get download url
        if file_id is None:
            file_id = self.get_file_ids(filename=filename, 
                                        unique=True,
                                        exact_match=True)[0]
        fields = self._service.files().get(fileId=file_id, fields='webContentLink, size').execute()
        size = int(fields['size'])
        #download_url = fields['webContentLink'] # doesn't work
        download_url = self.BASE_DOWNLOAD_URI + '/' + file_id + '?alt=media'
        #download_url = self._service.files().get_media(fileId=file_id).uri # also works
        
        # Set range of bytes to download
        if byte_range is None:
            byte_range = [0, size-1]
        
        # Download and return all bytes at once if no file object provided
        if file_object is None:
            
            header = {'Range': 'bytes=%d-%d' % (byte_range[0], byte_range[1])}
            resp, content = self._service._http.request(download_url, headers=header)
            
            return content
        
        # Otherwise, write chunks to file object until all bytes downloaded
        else:
            pointer = byte_range[0]
            while pointer < byte_range[1]:
                # Get next chunk and write to file object
                end = min(pointer + chunk_size - 1, size - 1)
                header = {'Range': 'bytes=%d-%d' % (pointer, end)}
                resp, content = self._service._http.request(download_url, headers=header)
                file_object.write(content)

                # Print progress update
                if verbose and (self._progress(end+1, byte_range[1]) > self._progress(pointer, byte_range[1])):
                    print('%d%% complete...' % self._progress(end+1, byte_range[1]))

                # Update pointer
                pointer += chunk_size
            
            return True

    def upload(self, *,
               filepath=None,
               file_stream=None,
               filename=None,
               folder_ids=None,
               mime_type='application/octet-stream',
               chunk_size=1024*256*4,
               verbose=True):
        """
        Upload a file to Google Drive.

        Args:
        - filepath: Filepath on local disk of file to be uploaded.
        - filename: Name to be given to uploaded file.
        - folder_ids: List of ID(s) of folder(s) to upload file to.
        - mime_type: MIME type of upload file.
        - chunk_size: Size in bytes of each chunk to send to server.
        - verbose: If True, print upload progress.

        Returns:
        - file_metadata: Dict containing metadata of uploaded file.

        Useful resources:
        - Google Drive API upload help: 
          https://developers.google.com/drive/api/v3/manage-uploads
        - Google Drive API http source code:
          https://github.com/googleapis/google-api-python-client/blob/master/googleapiclient/http.py#L872
        - httplib2 help (e.g. self._service._http):
          https://httplib2.readthedocs.io/en/latest/libhttplib2.html#http-objects
        """
        # Check parameters
        if (chunk_size % 1024*256 != 0):
            raise SyntaxError('chunk_size must be multiple of 256 KB.')
        
        # Create byte I/O stream
        if filepath is not None:
            f = open(filepath, 'rb')
        elif file_stream is not None:
            f = file_stream
        else:
            raise SyntaxError('Filepath or I/O object must be provided.')
        f.seek(0, os.SEEK_END)
        total_size = f.tell() # size of file in bytes

        # Get Drive metadata
        if filename is None:
            filename = filepath.split('/')[-1]
        if folder_ids is None:
            folder_ids = [] # store in MyDrive
        elif not isinstance(folder_ids, list):
            folder_ids = [folder_ids]

        # Initiate resumable session by uploading file metadata
        file_metadata = {'name': filename,
                        'parents': folder_ids}
        body = json.dumps(file_metadata)
        headers = {'X-Upload-Content-Type': mime_type,
                   'X-Upload-Content-Length': str(total_size),# size of entire file,
                   'Content-Type': 'application/json; charset=UTF-8', # use metadata
                   'Content-Length': str(len(body)) # size of metadata
                  }
        init_uri = self.BASE_UPLOAD_URI + '?uploadType=resumable'
        response, content = self._service._http.request(init_uri, 
                                                        method='POST',
                                                        body=body,
                                                        headers=headers)
        
        # Get resumable upload URI if session successfully initiated
        if response['status'] == '200':
            upload_uri = response['location']
        else:
            self._raise_http_error(response, content)

        # Upload in chunks until complete
        start = 0
        while (start == 0) or (response['status'] == '308'):
            # Get next chunk
            f.seek(start, os.SEEK_SET) # set pointer
            chunk = f.read(chunk_size) # read next chunk
            chunk_size_ = len(chunk) # if chunk is less than max size
            end = start + chunk_size_ - 1

            # Upload chunks to upload uri
            headers = {'Content-Range': 'bytes %d-%d/%d' % (start, end, total_size),
                       'Content-Length': str(chunk_size_)}
            response, content = self._service._http.request(upload_uri, 
                                                            method='PUT',
                                                            body=chunk,
                                                            headers=headers)

            # Print progress update
            if verbose and (self._progress(end+1, total_size) > self._progress(start, total_size)):
                print('%d%% complete...' % self._progress(end+1, total_size))
            
            # Update pointer index
            if 'range' in response:
                start = int(response['range'].split('-')[-1]) + 1
            else:
                start = end + 1
        
        # Handle outcomes
        if response['status'] in ['200', '201']:
            print('File sucessfully uploaded.')
            content = json.loads(content.decode('utf-8'))
            #print('File ID: {}'.format(content['id']))
            #print('Filename: {}'.format(content['name']))
            #print('MIME Type: {}'.format(content['mimeType']))
            return content
        else:
            self._raise_http_error(response, content)

    def _progress(self, current, total, percentile=10):
        """Returns current progress rounded down to nearest percentile."""
        return (100*(current/total) // percentile)*percentile

    def _raise_http_error(self, response, content):
        raise HTTPError(response['status'], content.decode('utf-8'))

    def _make_request(self, request, timeout=5.0, max_attempts=10):
        """Make HTTP request repeatedly, with timeouts in-between"""
        attempt = 0
        while attempt < max_attempts:
            try:
                return request.execute()
            except HttpError as e:
                time.sleep(timeout)
        
        

    def _get_file_ids(self, *,
                      filename=None, 
                      mime_type=None,
                      exact_match=False,
                      parent=None):
        """
        Search for file IDs given criteria. Remember that folders are considered
        files with the MIME type 'application/vnd.google-apps.folder'.

        Useful resources:
        - https://developers.google.com/drive/api/v3/search-files
        """
        q = ""
        
        # Query filename
        if filename is not None:
            if exact_match:
                q_rel = '='
            else:
                q_rel = 'contains'
            q += "name {} '{}'".format(q_rel, filename)

        # Query MIME type
        if mime_type is not None:
            if len(q) > 0:
                c = ' and '
            else:
                c = ''
            q += "{}mimeType = '{}'".format(c, mime_type)

        # Query parents
        if parent is not None:
            # Parent must be string if either name or ID provided
            assert isinstance(parent, str)
            try: # convert parent name to folder ID
                parents = self.get_folder_ids(foldername=parent, exact_match=True)
                #parent = ','.join(parent)
            except SyntaxError: # assume parent ID provided
                parents = [parent]

            # In case of multiple folders matching parent name, 
            # create list of parent queries
            p = "("
            for parent in parents[:-1]:
                p += "'{}' in parents or ".format(parent)
            p += "'{}' in parents)".format(parents[-1])
            
            # Add to previous query
            if len(q) > 0:
                c = ' and '
            else:
                c = ''
            q += "{}{}".format(c, p)

        # Query for files
        if len(q) > 0:
            files = self._service.files().list(q=q).execute()['files']
        else:
            raise SyntaxError('Empty query string.')
        
        # Return if not empty
        if len(files) == 0:
            raise SyntaxError('No files matching \'%s\'.' % filename) 
        return [f['id'] for f in files]


    def get_file_ids(self, *, 
                     filename=None, 
                     unique=False, 
                     mime_type=None,
                     exact_match=False,
                     parent=None):
        # Get file IDs containing filename
        file_ids = self._get_file_ids(filename=filename, 
                                      mime_type=mime_type,
                                      exact_match=exact_match,
                                      parent=parent)
        
        if (len(file_ids) > 1) and unique:
            raise SyntaxError('Multiple files matching \'%s\'. Please specify.' % filename)
        else:
            return file_ids

    def _get_file_metadata(self, file_ids, fields=None):
        # Convert to list if needed
        if not isinstance(file_ids, list):
            file_ids = [file_ids]
        
        # Format field parameter (loosely)
        if isinstance(fields, list):
            fields = ','.join(fields)

        # Get file metadata
        files = []
        for file_id in file_ids:
            request = self._service.files().get(fileId=file_id, fields=fields)
            metadata = self._make_request(request)
            files.append(metadata)
        return files

    def get_file_metadata(self, *,
                          filename=None,
                          file_ids=None, 
                          fields=None, 
                          unique=False,
                          parent=None):
        # Get file metadata from list of file IDs
        if filename is not None:
            file_ids = self.get_file_ids(filename=filename, 
                                         unique=unique,
                                         parent=parent)
        return self._get_file_metadata(file_ids, fields=fields)
 
    def get_folder_ids(self, *,
                       foldername=None,
                       filename=None, 
                       unique=False,
                       exact_match=False,
                       parent=None):
        if foldername is not None:
            # Get IDs of folder(s) with foldername
            return self.get_file_ids(filename=foldername,
                                     unique=unique,
                                     mime_type='application/vnd.google-apps.folder',
                                     exact_match=exact_match,
                                     parent=parent)
            
        elif filename is not None:
            # Get file ID(s) of filename
            file_ids = self.get_file_ids(filename=filename, 
                                         unique=unique,
                                         exact_match=exact_match)

            # Create list of all folder IDs containing file ID(s)
            folder_ids = []
            for file_id in file_ids:
                parents = drive_service._service.files().get(fileId=file_id, fields='parents').execute()['parents']
                folder_ids += [p for p in parents]

            return folder_ids
        
        else:
            raise SyntaxError('Foldername or filename must be provided.')

class GoogleDriveFile:

    def __init__(self, service, filename):
        # Set global attributes
        self._service = service # GoogleDriveService object
        self.filename = filename
        self.file_id = self._service.get_file_ids(filename, unique=True)


        # Set initial parameters
        self.pointer = 0 # pointer to current byte

    def seek(self, idx, code):
        """Wrapper for Python file.seek"""
        if code == 0:
            self.pointer = idx
        else:
            raise ValueError('Unknown code %d.' % code)

        return self.pointer

    def read(self, num_bytes):
        """Wrapper for Python file.read"""
        chunk = self._service.download(file_id=self.file_id, 
                                       byte_range=[self.pointer, self.pointer+num_bytes-1])
        self.pointer += num_bytes

        return chunk

    def close(self):
        self.pointer = 0


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

    def __init__(self, file_object):
        """
        Reads MDA files generated by MountainSort software. Assumes data shape is
        [num_channels, num_samples].

        Args:
        - file_object: Python file-object in read-binary mode; must have seek() 
                       and read() methods (e.g. from built-in function 
                       open(filename, 'rb'))
        """
        self.f = file_object
        self.header = MDAHeader(f=self.f) # Shape values
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

def bytes_to_object(chunk, ob_type='numpy', load_fn=None):
    """Loads bytes into specified object by creating temporary file.
    
    Args:
    - chunk: chunk of binary data to load
    - ob_type: type of object to create
        - numpy: numpy array
        - json: JSON file
    - load_fn: specific function to interpret byte object if needed.
    """
    # Load temporary file-like object
    temp = TemporaryFile()
    temp.write(chunk)
    temp.seek(0, 0)

    # Create object
    if ob_type == 'numpy':
        if load_fn == None:
            obj = np.load(temp)
        else:
            obj = load_fn(temp)

    elif ob_type == 'json':
        if load_fn == None:
            obj = json.load(temp)
        else:
            obj = load_fn(temp)
    else:
        raise SyntaxError('Unknown object type \'%s\'.' % ob_type)

    # Close temporary file
    temp.close()

    return obj

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
                break
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
    """
    Determine which time points in a vector lie within the 
    half-open interval [t1, t2).

    Args:
    - t (ndarray): 1D array containing time points to query.
    - t1 (ndarray): 1D array containing time points defining interval starts.
    - t2 (ndarray): 1D array containing time points defining interval ends.
    - query (str): Specifies the type of information to return.
        - 'event': Returns 1D array of len(t) in which the ith entry corresponds 
            to the number of intervals containing t[i].
        - 'interval': Returns 1D array of len(t1) in which the ith entry
            corresponds to the number of events in t contained in the ith interval.
        - 'array': Returns 2D array in which the [i, j] entry is True if t[i]
            is contained in the jth interval.
        - 'event_interval': Returns 1) 1D array containing event indices and 2)
            1D array of which interval(s) contain the corresponding event indices 
    """
    gt_t1 = (t[np.newaxis, :] >= t1[:, np.newaxis])
    lt_t2 = (t[np.newaxis, :] < t2[:, np.newaxis])
    bt_t1_t2 = np.logical_and(gt_t1, lt_t2)

    if query == 'event':
        return np.sum(bt_t1_t2.astype(np.int32), axis=0)
    elif query == 'interval':
        return np.sum(bt_t1_t2.astype(np.int32), axis=1)
    elif query == 'array':
        return bt_t1_t2
    elif query == 'event_interval':
        return np.argwhere(bt_t1_t2)[:, 1], np.argwhere(bt_t1_t2)[:, 0]
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
        for i, (a_i, idx) in enumerate(zip(a, ids)):
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
    
    # Filter out NaN values
    if ignore_nan:
        idx = ~np.isnan(stats_all)
        stats_all = stats_all[idx]
        ids_all = ids_all[idx]

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
        if np.sum(idx_) > 0:
            stats_id[i] = f(stats_all[idx_])
            for i in range(n_args):
                args_id[i].append(args_all[i][idx_])
        else:
            stats_id[i] = np.array([np.nan])
            for i in range(n_args):
                args_id[i].append([])
    
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

def get_peristimulus_data(data, 
                          t_data, 
                          t_stimulus, 
                          window=[-1.0, 1.0], 
                          fs=1.0):
    # Get traces corresponding to stimulus windows
    idx_window = np.arange(round(window[0]*fs), round(window[1]*fs)).astype(np.int64)
    data_win =  []
    for i in range(t_stimulus.shape[0]):
        # Get trace around stimulus
        idx_center = np.squeeze(np.argwhere(np.isclose(t_data, t_stimulus[i], atol=1.0/fs)))
        if idx_center.size == 0:
            continue
        elif idx_center.size > 1:
            idx_center = idx_center[idx_center.size//2]
        idx_i = idx_window + idx_center
        data_win.append(data[idx_i])
    data_win = np.array(data_win)

    # Get corresponding timestamps relative to window
    t_window = idx_window / fs

    return t_window, data_win

def dec_to_bin_array(d, bits=None, bit_order='<'):
    # No need for axis argument: just put newaxis on end,
    # and can transpose later if needed
    
    # Convert to numpy array if needed
    if not isinstance(d, np.ndarray):
        d = np.array([d])
    
    # Calculate minimum bits to represent integer
    min_bits = np.max(np.ceil(np.log2(np.maximum(d, np.ones(d.shape))))).astype(np.int64)
    if bits is None:
        bits = min_bits
    elif bits < min_bits:
        raise ValueError('Not enough bits to represent number. At least %d bits required.' % min_bits)
    
    # Broadcast binary representation to new axis at end
    b = np.zeros(list(d.shape) + [bits], dtype=np.int16)
    rem = d
    bit = bits
    while (bit > 0) and (rem > 0).any():
        bit -= 1
        b[..., bit] = (rem >= 2**bit)
        rem = rem % 2**bit
    
    # Return binary representation
    if bit_order == '<':
        return b
    elif bit_order == '>':
        return b[..., ::-1]
    else:
        raise ValueError('Unknown bit order \'%s\'' % bit_order)


### Media Objects ###
import subprocess as sp
import multiprocessing as mp
import tempfile
import shutil
import time
import glob
from matplotlib.ticker import NullLocator
try:
    import imageio
except ModuleNotFoundError as e:
    print(e, 'Some media functions may not be available.')

class VideoAnnotator:
    
    def __init__(self,
                 input_filename,
                 annotate_fn,
                 annotate_kwargs={},
                 output_filename=None,
                 output_resolution=None,
                 fps=None,
                 chunk_size=60,
                 frame_limit=100):
        # Name settings
        self.filename = input_filename
        self.video_format = input_filename.split('.')[-1]
        if output_filename is None:
            base_filename = ''.join(input_filename.split('.')[0:-1])
            self.output_filename = '.'.join([base_filename + '_annotated',
                                            self.video_format])
        elif not output_filename.endswith(self.video_format):
            raise ValueError('Output must be same format as input.')
        else:
            self.output_filename = output_filename
            
        # Create reader
        self.reader = VideoReader(input_filename)
        
        # Video settings
        if output_resolution is None:
            self.output_res = self.reader.get_metadata('resolution')
        else:
            self.output_res = output_resolution
        if fps is None:
            self.fps = self.reader.get_metadata('fps')
        else:
            self.fps = fps
        
        # Annotation settings
        self._annotate_fn = annotate_fn
        self._annotate_kwargs = annotate_kwargs

        # Runtime settings
        self.chunk_size = chunk_size # seconds
        self.frame_limit = frame_limit # max frames to load per worker
        
        # Get video duration in seconds
        self.duration = self.reader.get_metadata('duration')
        self.num_processes = int((self.duration // self.chunk_size) + (self.duration % self.chunk_size > 0))
        
    def run(self, max_workers=2, verbose=True):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize workers
        workers = [] # list of current jobs
        n = 0 # job number
        q = mp.Queue() # queue to store results
        for i in range(min(max_workers, self.num_processes)):
            p = mp.Process(target=self._process_chunk, 
                           args=(i, q), 
                           name='{}-{} seconds'.format(n*self.chunk_size, (n+1)*self.chunk_size))
            if verbose:
                print('starting process {}'.format(p.name))
            p.start()
            workers.append(p)
            n += 1
            
        while True:
            # Check for finished jobs
            for i, p in enumerate(workers):
                if p.exitcode is not None: # process finished
                    if verbose:
                        print('finished process {}'.format(p.name))
                    if n < self.num_processes:
                        # Delete finished process
                        del p
                        
                        # Add next process
                        p = mp.Process(target=self._process_chunk, 
                                       args=(n, q), 
                                       name='{}-{} seconds'.format(n*self.chunk_size, (n+1)*self.chunk_size))
                        if verbose:
                            print('starting process {}'.format(p.name))
                        p.start()
                        workers[i] = p
                        n += 1
                    else:
                        # Delete finished process and trim list
                        del workers[i]
            
            # Stop when no more jobs remain
            if len(workers) == 0:
                break
                
        # Get video chunk filenames and order
        p_order = []
        chunk_filenames = []
        while not q.empty():
            (i, name) = q.get()
            p_order.append(i)
            chunk_filenames.append(name)
        sort_idx = [i[0] for i in sorted(enumerate(p_order), key=lambda x: x[1])]
        
        # Stitch videos together
        # (see https://stackoverflow.com/a/11175851/8066298)
        list_filename = '/'.join([self.temp_dir, 'video_list.txt'])
        with open(list_filename, 'w') as f:
            for name in [chunk_filenames[i] for i in sort_idx]:
                f.write('file {}/{}'.format(self.temp_dir, name))
                f.write('\n')
                f.write('duration {:.2f}'.format(self.chunk_size))
                f.write('\n\n')
        cmd = ['ffmpeg',
               '-v', 'error',
               '-f', 'concat',
               '-safe', '0',
               '-i', list_filename,
               '-c', 'copy',
               '-s', '{}x{}'.format(*self.output_res),
               self.output_filename]
        with sp.Popen(cmd, stderr=sp.PIPE) as p:
            self._run_process(p)
        
        # Delete temporary directory when finished
        shutil.rmtree(self.temp_dir)

    def _process_chunk(self, n, q):
        # Create decoder for chunk
        t_start = n*self.chunk_size
        t_stop = t_start + self.chunk_size
        reader = VideoReader(self.filename,
                             t_start=t_start,
                             t_stop=t_stop,
                             max_frames=self.frame_limit)
        
        # Create encoder for chunk
        # (see https://stackoverflow.com/a/13298538/8066298)
        chunk_filename = 'video-{:04d}.{}'.format(n, self.video_format)
        cmd = ['ffmpeg',
               '-v', 'error',
               '-vcodec', 'mjpeg',
               '-f', 'image2pipe',
               '-i', '-',
               '-vcodec', 'mpeg4',
               '-framerate', '{:d}'.format(int(self.fps)),
               '-s', '{}x{}'.format(*self.output_res),
               '{}/{}'.format(self.temp_dir, chunk_filename)]
        writer = sp.Popen(cmd, stdin=sp.PIPE)
        
        # Iterate through decoded frames for annotation
        num_frames = int(self.chunk_size*self.fps)
        for i in range(num_frames):
            # Get frame
            frame = reader.next_frame()
            
            # Find global frame number
            n_frame = int(n*self.chunk_size*self.fps + i)
            
            # Annotate frame
            fig = self._annotate_frame(n_frame, frame)
            
            # Save annotated frame to encoder stdin
            fig.savefig(writer.stdin,
                        format='JPEG',
                        bbox_inches='tight', 
                        pad_inches=0, 
                        dpi=96)

            # IMPORTANT: Manually garbage collect to release memory
            fig.clf()      # clear figure
            plt.close(fig) # close figure
            gc.collect()   # force garbage collection
        
        # Check to make sure no frames are left in segment
        if reader.next_frame() is not None:
            raise UserWarning('Some frames may be skipped in chunk {}.'.format(n))
        reader.close()

        # Place video filename in queue
        q.put([n, chunk_filename])
        
        # Wait for encoder to finish
        writer.stdin.close()
        writer.wait()
            
    def _run_process(self, p):
        c = p.wait()
        if c > 0:
            print(p.stderr.read(10000).decode())
            raise RuntimeError('The subprocess {} exited with return code {}.'
                               .format(' '.join(p.args), c))
    
    def _annotate_frame(self, n_frame, frame):
        # Get frame dimensions
        #height, width, channel = frame.shape
        width, height = self.output_res
        
        # Create matplotlib figure from image data
        # (for figure sizes, see https://stackoverflow.com/a/13714720/8066298)
        fig, ax = plt.subplots(figsize=(width/96, height/96), dpi=96)
        ax.imshow(frame)
        
        # Annotate frame as specified
        self._annotate_fn(n_frame, ax, **self._annotate_kwargs)
        
        # Remove borders
        # (see https://stackoverflow.com/a/27227718/8066298)
        ax.set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, 
                            hspace=0, wspace=0)
        plt.margins(0,0)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        
        # Return annotated figure
        return fig


class VideoReader:
    # Improvements:
    # - Change max_frames to max_memory by estimating how much memory each frame will use
    
    def __init__(self,
                 input_filename,
                 t_start=0.0,
                 t_stop=None,
                 max_frames=100 # max number of frames to load at a time
                ):
        # Settings
        self.input_filename = input_filename
        self.t_start = t_start
        if t_stop is None:
            self.t_stop = self.get_metadata('duration')
        else:
            self.t_stop = t_stop
        
        # Frame attributes
        self._width = self._resolution[0]
        self._height = self._resolution[1]
        self._channels = 3 # hard-coded; see +rgb24 flag in _load_frames()
        self.frame_size = self._width*self._height*self._channels
            
        # Memory mangagement
        self.max_frames = max_frames
        self._sp = None
        
        # Compute time chunk size to load. Round down to nearest half second
        # in order to avoid rounding errors while seeking. (Remember that 
        # FFMPEG seeks by time, not frame number.)
        eps = 0.5 # decimal to round down to
        self._dt_chunk = int(self.max_frames/self._fps/eps)*eps
        self._t = self.t_start # pointer to current time
    
    def get_metadata(self, query):
        if query.lower() == 'duration':
            return self._duration
        elif query.lower() == 'resolution':
            return self._resolution
        elif query.lower() == 'fps':
            return self._fps
    
    @property
    def _duration(self):
        cmd = ['ffprobe',
               '-v', 'error',
               '-i', self.input_filename,
               '-select_streams', 'v',
               '-show_entries', 'format=duration',
               '-of', 'csv=p=0']
        return float(self._run_metadata_command(cmd))
    
    @property
    def _resolution(self):
        cmd = ['ffprobe',
               '-v', 'error',
               '-i', self.input_filename,
               '-select_streams', 'v',
               '-show_entries', 'stream=width,height',
               '-of', 'csv=p=0']
        s = self._run_metadata_command(cmd).strip(' \n')
        return [int(d) for d in s.split(',')]
    
    @property
    def _fps(self):
        cmd = ['ffprobe',
               '-v', 'error',
               '-i', self.input_filename,
               '-select_streams', 'v',
               '-show_entries', 'stream=avg_frame_rate',
               '-of', 'csv=p=0']
        s = self._run_metadata_command(cmd).strip(' \n')
        return float(s.split('/')[0]) / float(s.split('/')[1])
             
    def _run_metadata_command(self, cmd, bufsize=100):
        with sp.Popen(cmd, bufsize=bufsize, stdout=sp.PIPE, stderr=sp.PIPE) as p:
            p.wait() # waits until process has terminated
            s = p.stdout.read(bufsize).decode()
            
        return s
    
    def _run_pipe_command(self, cmd):
        # Begin running process, but do not wait for it to finish
        return sp.Popen(cmd,
                        bufsize=None,
                        stdout=sp.PIPE,
                        stderr=sp.PIPE)

    def next_frame(self):
        # Check to see if video loaded in memory
        if self._sp is None:
            self._load_frames()
        
        # Read next frame from bytes
        frame = self._sp.stdout.read(self.frame_size)
        
        if len(frame) > 0:
            # Return next frame if pipe not empty 
            frame = np.frombuffer(frame, dtype='uint8')
            return frame.reshape([self._height, self._width, self._channels], order='C')
        elif self._t < self.t_stop:
            # Load next set of frames if pipe empty
            self._load_frames()
            return self.next_frame()
        else:
            # Return null object if reached end of video
            return None
        
    def _load_frames(self):
        """Loads video frames into memory by directing ffmpeg output to stdout of subprocess."""
        # Check for end of video
        if self._t >= self.t_stop:
            raise RuntimeError('Reached end of video segment.')
        
        # Close current pipe
        self._close_sp()
                
        # Begin piping output to stdout
        dt_chunk = min(self._dt_chunk, self.t_stop - self._t)
        #pad = 1.00 # pad to help accurate seeking
        cmd = ['ffmpeg',
               '-v', 'error',
               #'-ss', '{:.4f}'.format(self._t - pad),
               '-ss', '{:.4f}'.format(self._t),
               '-i', self.input_filename,
               #'-ss', '{:.4f}'.format(pad - 0.01),
               '-t', '{:.4f}'.format(dt_chunk),
               '-f', 'image2pipe',
               '-pix_fmt', '+rgb24', # throw error if rgb24 unavailable
               '-vcodec', 'rawvideo', # stream frame pixels only
               '-' # pipe to stdout
              ]
        self._sp = self._run_pipe_command(cmd)
        
        # Increment time pointer
        self._t += dt_chunk

    def __enter__(self):
        return self

    def __exit__(self):
        self._close_sp()

    def close(self):
        self._close_sp()

    def _close_sp(self):
        if self._sp is not None:
            try:
                outs, errs = self._sp.communicate(timeout=10)
            except TimeoutExpired:
                self._sp.kill()
                outs, errs = self._sp.communicate()