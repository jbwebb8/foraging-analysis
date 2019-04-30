### Functions for future ephys-analysis repo ###
import json
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm_notebook
import util

import numpy as np
#try:
#    import cupy as np
#    print('cupy successfully imported.')
#except ImportError as e:
#    import numpy as np
#    print('Error importing cupy. Defaulting to numpy.')

try:
    from mountainlab_pytools.mdaio import writemda64
    from mountainlab_pytools import mlproc as mlp
except ModuleNotFoundError as e:
    print('mountainlab_pytools module not installed. Some functions'
          ' from the ephys package may not be available.')


### MountainSort functions ###
DEFAULT_SORT_PARAMS = {
    'processor_names': ['all'],
    'output_dir': './',
    'geom_filepath': '',
    'sample_rate': 30000,
    'freq_min': 300,
    'freq_max': 6000,
    'adjacency_radius': -1,
    'detect_sign': -1,
    'detect_threshold': 4,
    'bursting_parents': True,
    'clip_size': 100,
    'refrac_msec': 1.0,
    'firing_rate_thresh': 0.1,
    'isolation_thresh': 0.95,
    'noise_overlap_thresh': 0.03,
    'peak_snr_thresh': 1.5,
    'opts': {}
}
# Note: mlp.addProcess takes parameters of the form
# (processor_name, input_dict, output_dict, params_dict, opts_dict)
def sort_spikes(*,
                timeseries,
                config=None,
                **kwargs):
    # Inital time for calculating elapsed time
    t_start = time.time()

    # Load sort parameters
    print('Loading parameters...', end=' ')
    if config is not None:
        if isinstance(config, str): # JSON filename
            user_params = util.load_json(config)
        elif isinstance(config, dict):
            user_params = config
    else:
        user_params = kwargs
    
    params = {}
    for k, v in DEFAULT_SORT_PARAMS.items():
        if k in user_params.keys():
            params[k] = user_params[k]
        else:
            params[k] = DEFAULT_SORT_PARAMS[k]
    print('done.')

    # Annoying bug: booleans must be lowercase strings
    util.recursive_dict_search(params, True, 'true')
    util.recursive_dict_search(params, False, 'false')

    # File settings
    output_dir = params['output_dir']
    base_fn = timeseries[timeseries.rfind('/')+1:timeseries.find('.mda')]
    base_fp = output_dir + base_fn
    
    ### Create jobs ###
    all_jobs = ('all' in params['processor_names'])

    # Bandpass filter
    processor_name = 'ephys.bandpass_filter'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'timeseries': timeseries},
                    {'timeseries_out': base_fp + '_filt.mda'},
                    {'samplerate': params['sample_rate'],
                        'freq_min': params['freq_min'],
                        'freq_max': params['freq_max']},
                    params['opts'])

    # Whiten
    processor_name = 'ephys.whiten'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'timeseries': base_fp + '_filt.mda'},
                    {'timeseries_out': base_fp + '_whitened.mda'},
                    {},
                    params['opts'])

    # Sort
    processor_name = 'ms4alg.sort'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'timeseries': base_fp + '_whitened.mda',
                        'geom': params['geom_filepath']},
                    {'firings_out': output_dir + 'firings.mda'},
                    {'adjacency_radius': params['adjacency_radius'],
                        'detect_sign': params['detect_sign'],
                        'detect_threshold': params['detect_threshold']},
                    params['opts'])
    
    # Calculate cluster metrics
    processor_name = 'ms3.cluster_metrics'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess('ms3.cluster_metrics',
                    {'firings': output_dir + 'firings.mda',
                        'timeseries': base_fp + '_whitened.mda'},
                    {'cluster_metrics_out': output_dir + 'cluster_metrics_ms3.json'},
                    {'samplerate': params['sample_rate']},
                    params['opts'])

    processor_name = 'ms3.isolation_metrics'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'firings': output_dir + 'firings.mda',
                        'timeseries': base_fp + '_whitened.mda'},
                    {'metrics_out': output_dir + 'isolation_metrics.json',
                        'pair_metrics_out': output_dir + 'pair_metrics.json'},
                    {'compute_bursting_parents': params['bursting_parents']},
                    params['opts'])

    processor_name = 'ephys.compute_cluster_metrics'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'firings': output_dir + 'firings.mda',
                        'timeseries': base_fp + '_whitened.mda'},
                    {'metrics_out': output_dir + 'cluster_metrics_ms4.json'},
                    {'samplerate': params['sample_rate'],
                        'clip_size': params['clip_size'],
                        'refrac_msec': params['refrac_msec']},
                    params['opts'])

    processor_name = 'ms3.combine_cluster_metrics'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'metrics_list': [output_dir + 'cluster_metrics_ms3.json', 
                                      output_dir + 'isolation_metrics.json',
                                      output_dir + 'cluster_metrics_ms4.json']},
                    {'metrics_out': output_dir + 'metrics.json'},
                    {},
                    params['opts'])

    # Add curation tags (but don't reject)
    processor_name = 'pyms.add_curation_tags'
    if processor_name in params['processor_names'] or all_jobs:
        mlp.addProcess(processor_name,
                    {'metrics': output_dir + 'metrics.json'},
                    {'metrics_tagged': output_dir + 'metrics_tagged.json'},
                    {'firing_rate_thresh': params['firing_rate_thresh'],
                        'isolation_thresh': params['isolation_thresh'],
                        'noise_overlap_thresh': params['noise_overlap_thresh'],
                        'peak_snr_thresh': params['peak_snr_thresh']},
                    params['opts'])

def track_pipeline_progress(mlclient):
    ### Track progress ###
    t_start = time.time()
    job_times = {}
    status_old = _get_sort_status(mlclient)
    print(status_old)
    while len(status_old['pending']) + len(status_old['running']) > 0:
        # Check for changes in job status
        status_new = _get_sort_status(mlclient)
        updates = _get_sort_progress(status_old, status_new)
        print(status_old)
        print(status_new)

        # Print updates (if any)
        for name in updates['start']:
            print('Started %s...' % name)
            job_times[name] = [time.time()]
        for name in updates['finish']:
            t_1 = job_times[name][0]
            t_2 = time.time()
            job_times[name].append(t_2)
            h, m, s = util.format_elapsed_time(t_2 - t_1)
            print('Finished %s (time: %02d:%02d:%02.2f)' % (name, h, m, s))

        # Update current job statuses
        status_old = status_new
        time.sleep(1)
    
    # Print total elapsed time
    h, m, s = util.format_elapsed_time(time.time() - t_start)
    print('Finished script. (total time: %02d:%02d:%02d)' % (h, m, round(s)))

def _get_sort_status(mlclient):
    d = {'pending': [],
         'running': [],
         'finished': []}
    for job_id, job in mlclient._jobs.items():
        if job['status'] == 'pending':
            d['pending'].append(job['processor_name'])
        elif job['status'] == 'running':
            d['running'].append(job['processor_name'])
        elif job['status'] == 'finished':
            d['finished'].append(job['processor_name'])
    
    return d

def _get_sort_progress(status_old, status_new):
    d = {'start': [],
         'finish': []}
    for job_name in status_old['pending']:
        if job_name in status_new['running']:
            d['start'].append(job_name)
    for job_name in status_old['running']:
        if job_name in status_new['finished']:
            d['finish'].append(job_name)
    
    return d

def create_merge_map(*, firings_old, firings_new, merge_map_out=None):
    """
    Maps cluster labels from before to after curation step to track merges.
    """
    # Load firing data if filepaths provided
    if isinstance(firings_old, str):
        firings_old_filepath = firings_old
        with open(firings_old, 'rb') as f:
            firings_old = util.MDAReader(f).read()
    else:
        firings_old_filepath = ''
    if isinstance(firings_new, str):
        firings_new_filepath = firings_new
        with open(firings_new, 'rb') as f:
            firings_new = util.MDAReader(f).read()
    else:
        firings_new_filepath = ''

    
    # Iterate through each new cluster
    merge_map = []
    cluster_labels = np.unique(firings_new[2, :])
    for label in cluster_labels:
        s_firing = firings_new[1, np.where(firings_new[2, :] == label)]
        old_label = np.unique(firings_old[2, np.isin(firings_old[1, :], s_firing)])
        old_label = [l for l in old_label] # convert to list
        merge_dict = {'old_label': old_label,
                      'new_label': label}
        merge_map.append(merge_dict)
    
    # Create JSON file
    merge_map = {'clusters': merge_map,
                 'filenames': {'firings_old': firings_old_filepath,
                               'firings_new': firings_new_filepath} }
    if merge_map_out is not None:
        with open(merge_map_out, mode='w') as f:
            f.write(json.dumps(merge_map, indent=4, sort_keys=True))

    return merge_map


def update_cluster_tags(*, metrics_old, metrics_new, metrics_out=None):
    """
    Transfers cluster tags that were added during curation (to the 
    pre-curation metrics file) to the re-calculated metrics file.
    """
    # Load metrics if filepaths provided
    if isinstance(metrics_old, str):
        with open(metrics_old) as f:
            metrics_old = json.loads(f.read())
    if isinstance(metrics_new, str):
        with open(metrics_new) as f:
            metrics_new = json.loads(f.read())
    
    # Get numpy arrays of labels for faster processing
    labels_old = np.zeros(len(metrics_old['clusters']), dtype=np.int32)
    for i, metric in enumerate(metrics_old['clusters']):
        labels_old[i] = metric['label']
        
    labels_new = np.zeros(len(metrics_new['clusters']), dtype=np.int32)
    for i, metric in enumerate(metrics_new['clusters']):
        labels_new[i] = metric['label']
    
    # Transfer tags
    for i, metric in enumerate(metrics_new['clusters']):
        # Get tags from old labels
        idx_old = np.where(labels_old == metric['label'])[0]
        tags = []
        for j in idx_old:
            if 'tags' in metrics_old['clusters'][j].keys():
                tags += metrics_old['clusters'][j]['tags']
        
        # Add tags to new label
        if 'tags' not in  metrics_new['clusters'][i]:
            metrics_new['clusters'][i]['tags'] = []
        metrics_new['clusters'][i]['tags'] += tags
    
    # Create JSON file
    if metrics_out is not None:
        with open(metrics_out, mode='w') as f:
            f.write(json.dumps(metrics_new, indent=4, sort_keys=True))

    return metrics_new


def add_cluster_tags():
    pass


def get_templates(*, timeseries, 
                  firings,
                  templates_out=None,
                  window=[-1.0, 1.0], 
                  f_s=30000,
                  scale=1.0):
    # Load files if needed
    # firings can be loaded in memory, but timeseries usually too big
    if isinstance(timeseries, str):
        timeseries = util.MDAReader(open(timeseries, 'rb')) 
    if isinstance(firings, str):
        with open(firings, 'rb') as f:
            firings = util.MDAReader(f).read()

    # Set parameters
    cluster_labels = np.unique(firings[2, :])
    num_clusters = len(cluster_labels)
    win_wf = (np.array(window)/1000 * f_s).astype(np.int64) # ms --> samples
    win_len = np.diff(win_wf)[0]
    wfs = np.zeros([num_clusters, timeseries.num_channels, win_len])
    
    # Iterate through all units
    for i, label in enumerate(cluster_labels):
        t = time.time()
        print('Processing unit %d (%d of %d)...' % (label, i+1, len(cluster_labels)), end=' ')
        
        # Get spike sample indices
        idx_firings = firings[1, firings[2, :] == label].astype(np.int64)

        # Iterate through all spikes
        count = 0
        for j, idx in enumerate(idx_firings):
            #print('firing %d of %d' % (j+1, len(idx_firings)))
            idx_start = idx + win_wf[0]
            idx_end = idx + win_wf[1]

            # Do not add in edge effects
            if (idx_start >= 0) and (idx_end < timeseries.N):
                wfs[i, :, :] += timeseries.read(sample_start=idx_start, sample_end=idx_end)
                count += 1
        
        # Get mean for each channel over all spikes
        wfs[i, :, :] /= count # memory-efficient form of getting mean
        
        print('done. (%.3f seconds)' % (time.time() - t))
        
    # Convert to microvolts (see Intan data formats)
    wfs *= scale

    # Write to file
    if templates_out is not None:
        np.save(templates_out, wfs)

    return cluster_labels, wfs


def plot_templates(*, timeseries=None,
                   firings=None,
                   templates=None,
                   labels=None,
                   fig_out=None,
                   metrics=None,
                   geom=None,
                   plot_style='vertical',
                   window=[-1.0, 1.0],
                   f_s=30000,
                   fig_size=None,
                   **kwargs):
    # Get templates
    if (timeseries is not None) and \
       (firings is not None) and \
       (templates is None):
        labels, templates = get_templates(timeseries=timeseries,
                                          firings=firings,
                                          window=window,
                                          f_s=f_s,
                                          **kwargs)
    elif templates is not None:
        if isinstance(templates, str):
            with open(templates, 'rb') as f:
                templates = util.MDAReader(f).read()
    
    # Get labels
    if labels is not None:
        if isinstance(labels, str):
            labels = np.load(labels)
        tags = [[] for _ in range(len(labels))]
    if metrics is not None:
        if isinstance(metrics, str):
            with open(metrics, 'r') as f:
                metrics = json.loads(f.read())
        if labels is None:
            labels = [c['label'] for c in metrics['clusters']] 
            tags = [c['tags'] if 'tags' in c.keys() else [] for c in metrics['clusters']]
        else:
            tags = []
            for label in labels: # TODO: faster implementation
                for c in metrics['clusters']:
                    if c['label'] == label:
                        tags.append(c.get('tags', []))
    
    # Get geometry
    if geom is not None:
        if isinstance(geom, str):
            geom = np.loadtxt(geom) # rows of (x, z)
        channels = np.lexsort(geom.T) # sort by z, then x
    else:
        channels = np.arange(templates.shape[1]) # numerical order
    
    # Set parameters
    num_clusters = len(labels)#templates.shape[0]
    num_channels = len(channels)#templates.shape[1]
    N = templates.shape[2]
    t = np.linspace(window[0], window[1], N)

    # Create plot
    cols = min(len(labels), 4)
    rows = (num_clusters // cols) + (num_clusters % cols > 0)
    if fig_size is None:
        fig_size = (4*cols, num_channels/16*rows)
    fig, ax = plt.subplots(rows, cols, figsize=fig_size)

    for i, label in enumerate(labels):
        #idx = np.argwhere(labels == label).squeeze()
        idx = i # assume wfs in same order as labels
        X = templates[idx, :, :]
        if rows > 1 and cols > 1:
            ax_ = ax[i//cols][i%cols]
        elif rows > 1 or cols > 1:
            ax_ = ax[i]
        else:
            ax_ = ax

        if plot_style == 'vertical':
            # Rescale for aesthetics
            scale = 2.0 / np.max(np.abs(X))

            # Plot template for each channel
            for j, k in enumerate(channels):
                ax_.plot(t, scale*X[k, :] + j, color='black')
            
            # Plot t_0 (spike detection)
            ax_.vlines(t[np.searchsorted(t, 0.0)], ymin=0.0, ymax=num_channels,
                       linestyle='--', color='gray')
            
            # Plot settings
            tag_str = ', '.join(tags[i])
            if tag_str:
                tag_str = '(' + tag_str + ')'
            ax_.set_title('cluster %d\n%s' % (label, tag_str))
            ax_.set_xlabel('time (ms)')
            ax_.set_ylabel('channel')
            ax_.set_yticks(np.arange(num_channels))
            ax_.set_yticklabels(channels)

        elif plot_style == 'geometric':
            # Scale values to fit geometry
            x_range = np.max(geom[:, 0]) - np.min(geom[:, 0])
            x_scale = x_range / len(np.unique(geom[:, 0])) / np.max(np.abs(t))
            y_range = np.max(geom[:, 1]) - np.min(geom[:, 1])
            y_scale = y_range / len(np.unique(geom[:, 1])) / np.max(np.abs(X))

            # Plot waveform at location of channel
            for k, j in enumerate(channels):
                ax_.plot(x_scale*t + geom[k, 0], y_scale*X[k, :] + geom[k, 1], color='black')

            # Plot settings
            tag_str = ', '.join(tags[i])
            if tag_str:
                tag_str = '(' + tag_str + ')'
            ax_.set_title('cluster %d\n%s' % (label, tag_str))
            ax_.set_xlabel('x (um)')
            ax_.set_ylabel('z (um)')
            scalebar = AnchoredSizeBar(ax_.transData,
                                    1, '1 ms', 'lower right', 
                                    pad=0.1,
                                    color='black',
                                    frameon=False,
                                    size_vertical=1)
            ax_.add_artist(scalebar)

    # Configure plot
    plt.tight_layout()
    rem = (cols - len(labels) % cols) % cols
    for i in range(1, rem+1):
        if rows > 1:
            ax[-1, -i].axis('off')
        else:
            ax[-i].axis('off')

    if fig_out is not None:
        plt.savefig(fig_out)
        return True

    else:
        return fig, ax


def curate_firings(*, firings,
                   metrics,
                   keep_tags=[],
                   exclude_tags=[],
                   firings_out=None):
    # Load files if needed
    if isinstance(firings, str):
        with open(firings, 'rb') as f:
            firings = util.MDAReader(f).read()
    if isinstance(metrics, str):
        with open(metrics, 'r') as f:
            metrics = json.loads(f.read())
    
    # Make lists if needed
    keep_tags = util._check_list(keep_tags)
    exclude_tags = util._check_list(exclude_tags)

    # Get cluster metrics
    labels = np.array([c['label'] for c in metrics['clusters']])
    tags = [c['tags'] if 'tags' in c.keys() else [] for c in metrics['clusters']]
    
    # Create default keep indices
    if (len(keep_tags) > 0) and (len(exclude_tags) == 0):
        # If only keep criteria provided, default is exclude
        idx_keep = np.zeros(firings.shape[1], dtype=np.bool)
    elif (len(keep_tags) == 0) and (len(exclude_tags) == 0):
        # If only exclude criteria provided, default is keep
        idx_keep = np.ones(firings.shape[1], dtype=np.bool)
    else:
        # If both provided, default is keep 
        # (but exclude criteria take precedence)
        idx_keep = np.ones(firings.shape[1], dtype=np.bool)

    # Iterate through each cluster
    for i, [label, tags_i] in enumerate(zip(labels, tags)):
        # Get indices of cluster firings
        idx_firings = np.where(firings[2, :] == label)[0]
        
        # Update keep indices based on tags
        for tag in tags_i:
            if tag in keep_tags:
                idx_keep[idx_firings] = True
            elif tag in exclude_tags:
                idx_keep[idx_firings] = False
                break
    
    firings_curated = firings[:, idx_keep]

    if firings_out is not None:
        writemda64(firings_curated, firings_out)
        return True
    else:
        return firings_curated
        

### Firing rate estimation ###
def get_spike_counts(*,
                     firings,
                     t_firings,
                     t_stimulus,
                     dt_bin=0.020,
                     t_window=[-1.0, 1.0],
                     labels=None,
                     metrics=None,
                     verbose=True):
    """
    Returns number of spikes within time bins centered around time of stimuli.
    Equivalent to a peri-stimulus time histogram (PSTH).

    Args:
    - firings (ndarray, [3, N]): Array of spike information from curated MDA file.
    - t_firings (ndarray, [N]): Timestamps associated with each spike in firings.
    - t_stimulus (ndarray, [T]): Timestamps associated with onset of stimulus.
        Windows will be centered around these timestamps.
    - dt_bin (float): Size of time bins in seconds in which to count spikes.
    - t_window (list): Length of time window relative to t_stimulus in which
        to count spikes (of form [t_min, t_max]).
    - labels (ndarray, [M]): Cluster labels of units to be analyzed.
    - metrics (dict): Loaded JSON cluster metrics file from MountainSort.
    - verbose (bool): If True, print progress statements.

    Returns:
    - n_bins (ndarray, [M, T, num_bins]): Spike counts of M units across T stimuli
        within num_bins time bins in specified window around stimulus.
    - bins (ndarray, [num_bins+1]): Edges of time bins used to calculate spike counts relative to each
        time in t_stimulus.
    """
    # Create time bins within stimulus window
    bins = np.arange(t_window[0]/dt_bin, t_window[1]/dt_bin + dt_bin) * dt_bin
    t_bins = t_stimulus[:, np.newaxis] + bins[np.newaxis, :]

    
    # Get unit labels
    if labels is None:
        if metrics is not None:
            labels = np.array([c['label'] for c in metrics['clusters']])
        else:
            raise SyntaxError('Unit labels or metrics JSON file must be provided.')
    
    # Placeholder for binned spike counts
    # shape = [num_units, num_windows, bins]
    n_bin = np.zeros([len(labels), len(t_stimulus), len(bins)-1])

    # Iterate through units
    for i, label in enumerate(labels):
        if verbose:
            print('Processing unit %d (%d of %d)...' % (label, i+1, len(labels)), end=' ')
            t = time.time()

        # Get spike times
        t_firings_ = t_firings[firings[2, :] == label]

        # Determine number of spikes in bins over all stimulus windows
        in_bin = np.logical_and((t_firings_[np.newaxis, np.newaxis, :] >= t_bins[:, :-1, np.newaxis]), 
                                (t_firings_[np.newaxis, np.newaxis, :] <  t_bins[:, 1:, np.newaxis]))
        
        # Get spike counts in bins over all stimulus windows
        n_bin[i, :, :] = np.sum(in_bin.astype(np.int32), axis=-1) # sum over spikes

        if verbose:
            print('done. (%.3f seconds)' % (time.time() - t))
    
    return n_bin, bins


def smooth_spike_counts(*,
                        counts,
                        dt_bin,
                        method='kernel',
                        axis=-1,
                        **kwargs):
    """
    Provides continuous estimate of neural spike counts by smoothing histogram
    of spike counts in time bins. This can then be converted into firing rates
    by dividing the smoothed counts by dt_bin.

    Args:
    - counts (ndarray: Number of spikes in each time bin. Can be for single or
        multiple neurons (see axis parameter).
    - dt_bin (float): Size of time bins in seconds.
    - method (str): Smoothing method to use. Possible options are:
        - kernel: Apply kernel smoothing by convolving kernel with spike counts.
    - axis (int): Axis on which to apply smoothing. This should represent
        the axis containing the time bins (e.g. 2 if using get_spike_counts()).
    
    Returns:
    - n_smooth: Smoothed spike counts after applying specified method.
    """
    if method == 'kernel':
        return _kernel_smoothing(counts, dt_bin, axis=axis, **kwargs)
    else:
        raise SyntaxError('Unknown method \"%s\".' % method)


def _kernel_smoothing(counts,
                      dt_bin,
                      axis=0,
                      kernel=None,
                      kernel_type='Gaussian',
                      **kwargs):
    
    # Create kernel if not provided
    if kernel is None:
        kernel = _create_smoothing_kernel(kernel_type, dt_bin ,**kwargs)
        
    # Smooth counts by convolving with kernel
    n_smooth = _convolve(counts, kernel, axis=axis)
    
    return n_smooth
    

def _create_smoothing_kernel(kernel_type, dt_bin, **kwargs):
    if kernel_type == 'Gaussian':
        sigma = kwargs.get('sigma', 0.100) # kernel width (ms)
        sigma_k = sigma / dt_bin # kernel width (bins)
        return lambda x: 1/(2*math.pi*sigma_k**2)**0.5 * np.exp(-0.5 * x**2 / sigma_k**2)
    else:
        raise SyntaxError('Unknown kernel type "%s".' % kernel_type)

    
def _convolve(x, k, axis=0):
    if axis < 0:
        axis = x.ndim + axis
    a = axis
    b = x.ndim - axis - 1
    x_smooth = np.zeros(x.shape)
    for i in range(x.shape[axis]):
        slc = tuple([slice(None)]*a + [i] + [slice(None)]*b)
        idx = np.arange(x.shape[axis]) - i # zero-center mean
        k_i = k(idx)[tuple([np.newaxis]*a + [slice(None)] + [np.newaxis]*b)]
        x_smooth[slc] = np.sum(k_i * x, axis=axis)
    
    return x_smooth


def estimate_firing_rate(*,
                         firings,
                         t_firings,
                         t_trial,
                         t_window=[-1.0, 1.0],
                         labels=None,
                         metrics=None,
                         method='kernel',
                         **kwargs):
    pass


### Dimensionality reduction ###
def reduce_dimensions(Y,
                      method='PCA',
                      **kwargs):
    """
    Performs dimensionality reduction on spike counts or firing rates based
    on the specified method.

    Args:
    - Y (ndarray, [N, q, T]): Array containing the spike information. Note that
        it must be in the shape [# trials, # units, # time bins]. 
    - method (str): Dimensionality reduction method to apply. Options are:
        - 'pca'
        - 'ppca'
        - 'fa'
        - 'gpfa'
    
    Returns:
    - params
    - X
    """
    # Check input
    if Y.ndim != 3:
        raise SyntaxError('Y must be 3D array has %d dimensions.' % Y.ndim)
    else:
        # Placeholder for reference
        N = Y.shape[0] # number of trials
        q = Y.shape[1] # number of units
        T = Y.shape[2] # number of time bins

    # Apply dimensionality reduction method
    if method.lower() in ['pca', 'ppca', 'fa']:
        return _apply_two_step_method(Y, method, **kwargs)
    elif method.lower() == 'gpfa':
        return _apply_gpfa(Y, **kwargs)
    else:
        raise ValueError('Unknown method \'%s\'.' % method)

def _apply_two_step_method(Y,
                           method,
                           **kwargs):
    """
    """
    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins

    # Unroll into single series
    Y_rs = Y.transpose([1, 2, 0]).reshape([q, -1], order='F')

    # Calculate parameters and projections for given method
    params = _calculate_params(Y_rs, method, **kwargs)
    X = _calculate_projections(Y, method, **params, **kwargs)

    return params, X

def _apply_gpfa(Y, **kwargs):
    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins

    # Calculate parameters and projections
    params = _calculate_params(Y, 'gpfa', **kwargs)
    X = _calculate_projections(Y, 'gpfa', **kwargs)

    return params, X

def _calculate_params(Y, method, **kwargs):
    """
    """
    if method.lower() == 'pca':
        return _calculate_PCA_params(Y, **kwargs)
    elif method.lower() == 'ppca':
        return _calculate_PPCA_params(Y, **kwargs)
    elif method.lower() == 'fa':
        return _calculate_FA_params(Y, **kwargs)
    elif method.lower() == 'gpfa':
        return _calculate_GPFA_params(Y, **kwargs)
    else:
        raise ValueError('Unknown method \'%s\'.' % method)

def _calculate_projections(Y, method, **kwargs):
    """
    """
    if method.lower() == 'pca':
        return _calculate_PCA_projections(Y, **kwargs)
    elif method.lower() == 'ppca':
        return _calculate_PPCA_projections(Y, **kwargs)
    elif method.lower() == 'fa':
        return _calculate_FA_projections(Y, **kwargs)
    else:
        raise ValueError('Unknown method \'%s\'.' % method)

def _calculate_PCA_params(Y, p=1):
    """
    """
    # Shapes
    q = Y.shape[0] # number of units
    NT = Y.shape[1] # number of time bins * number of trials

    # Calculate eigenvectors of sample covariance matrix
    mu = np.mean(Y, axis=1, keepdims=True) # sample mean
    S = 1/NT * (Y - mu).dot((Y - mu).T) # sample covariance
    lam, U = np.linalg.eig(S) # eigendecomposition of sample covariance
    
    # Sort by eigenvalues
    sort_idx = np.argsort(lam)[::-1]
    lam = lam[sort_idx]
    U = U[:, sort_idx]

    # Keep first p dimensions
    lam = lam[:p]
    U = U[:, :p]

    # Package params
    params = {'U': U,
              'lamda': lam,
              'd': mu,
              'S': S}

    return params

def _calculate_PCA_projections(Y, *, U, d, 
                               p=None, 
                               **kwargs):
    """
    """
    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins

    # Set number of latent dimensions
    if p is None:
        p = U.shape[1]
    U_p = U[:, :p]

    # Calculate projections
    X = np.matmul(U_p.T, (Y - d)) # preserves dimension order

    return X

def _calculate_PPCA_params(Y, 
                           p=1, 
                           C_init=None, 
                           d_init=None, 
                           s_init=0.5, 
                           EM_steps=500,
                           orthonormal=True, 
                           verbose=False,
                           **kwargs):
    """
    """
    # Shapes
    q = Y.shape[0]
    NT = Y.shape[1]

    # Set initial values
    if C_init is None:
        C = np.random.rand(q, p)
    else:
        C = C_init
    if d_init is None:
        d = np.mean(Y, axis=1, keepdims=True)
    else:
        d = d_init
    s = s_init

    # Set progress bar
    if verbose:
        it = tqdm_notebook(range(EM_steps))
    else:
        it = range(EM_steps)

    # EM algorithm
    for i in it:
        # Cache old parameters
        C_old = C
        s_old = s

        # E-step: calculate E(x), E(xx^T) ~ new P(x|y)
        # (which maximizes expected joint probability wrt distribution of y)
        B = s * np.eye(p) + C.T.dot(C) # p x p
        A_inv = (1/s * np.eye(q)) - (1/s * C.dot(np.linalg.inv(B).dot(C.T))) # matrix inversion lemma
        E_x = C.T.dot(A_inv).dot(Y - d) # matrix form
        sum_E_xxT = ( (np.eye(p) - C.T.dot(A_inv.dot(C)))*NT
                        + E_x.dot(E_x.T) ) # summation form
        
        # M-step: update C and R to maximize likelihood
        # (which maximizes expected joint probability wrt parameters)
        C = ((Y - d).dot(E_x.T)).dot(np.linalg.inv(sum_E_xxT))
        s = 1.0/(NT*q) * np.trace( (Y - d).dot((Y - d).T) - C.dot(E_x.dot((Y - d).T)) )

        # Check for convergence
        if np.isclose(C, C_old).all() and np.isclose(s, s_old).all():
            if verbose:
                print('Converged after %d iterations.' % (i+1))
            break
        
    # Transform to othornormal space
    if orthonormal:
        U, D, VT = np.linalg.svd(C, full_matrices=False)
        D = np.diag(D)
    else:
        U, D, VT = None, None, None

    # Package params
    params = {'C': C,
              'd': d,
              's': s,
              'U': U,
              'D': D,
              'VT': VT}

    return params

def _calculate_PPCA_projections(Y, *, C, d, s, 
                                U=None, 
                                D=None, 
                                VT=None,
                                orthonormal=True,
                                **kwargs):

    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins

    # Set number of latent dimensions
    p = C.shape[1]

    # Calculate projections of X | Y
    # NOTE: matmul must be used instead of dot because Y.ndim > 2.
    # Additionally, shape is [N, q, T] because matmul broadcasts
    # over matrices residing in last two dimensions.
    B = s * np.eye(p) + C.T.dot(C) # p x p
    A_inv = (1/s * np.eye(q)) - (1/s * C.dot(np.linalg.inv(B).dot(C.T))) # matrix inversion lemma
    X = np.matmul(C.T, np.matmul(A_inv, (Y - d))) # matrix form

    # Linearly tranform x into othornomal space U
    if orthonormal:
        if (D is not None) and (VT is not None):
            X = np.matmul(D, np.matmul(VT, X))
        else:
            raise ValueError('Parameters for orthonormal transform (D, VT)'
                             ' must be provided.')
    
    return X

def _calculate_FA_params(Y,
                         p=1, 
                         C_init=None, 
                         d_init=None, 
                         R_init=None, 
                         EM_steps=500,
                         orthonormal=True, 
                         verbose=False,
                         **kwargs):
    """
    """
    # Shapes
    q = Y.shape[0]
    NT = Y.shape[1]

    # Set initial values
    if C_init is None:
        C = np.random.rand(q, p)
    else:
        C = C_init
    if d_init is None:
        d = np.mean(Y, axis=1, keepdims=True)
    else:
        d = d_init
    if R_init is None:
        R = np.diag(np.random.rand(q))
    else:
        R = R_init

    # Set progress bar
    if verbose:
        it = tqdm_notebook(range(EM_steps))
    else:
        it = range(EM_steps)

    # EM algorithm
    for i in it:
        # Cache old parameters
        C_old = C
        R_old = R

        # E-step: calculate E(x), E(xx^T) ~ new P(x|y)
        # (which maximizes expected joint probability wrt distribution of y)
        R_inv = np.diag(1.0 / R[np.arange(len(R)), np.arange(len(R))])
        B = np.eye(p) + C.T.dot(R_inv.dot(C)) # p x p
        A_inv = R_inv - R_inv.dot(C.dot(np.linalg.inv(B).dot(C.T.dot(R_inv)))) # matrix inversion lemma
        E_x = C.T.dot(A_inv).dot(Y - d) # matrix form
        sum_E_xxT = ( (np.eye(p) - C.T.dot(A_inv.dot(C)))*NT 
                    + E_x.dot(E_x.T) ) # summation form
        
        # M-step: update C and R to maximize likelihood
        # (which maximizes expected joint probability wrt parameters)
        C = ((Y - d).dot(E_x.T)).dot(np.linalg.inv(sum_E_xxT))
        R = 1/(NT) * np.eye(q) * ( (Y - d).dot((Y - d).T) - C.dot(E_x.dot((Y - d).T)) )

        # Check for convergence
        if np.isclose(C, C_old).all() and np.isclose(R, R_old).all():
            if verbose:
                print('Converged after %d iterations.' % (i+1))
            break

    # Transform to othornormal space
    if orthonormal:
        U, D, VT = np.linalg.svd(C, full_matrices=False)
        D = np.diag(D)
    else:
        U, D, VT = None, None, None

    # Package params
    params = {'C': C,
              'd': d,
              'R': R,
              'U': U,
              'D': D,
              'VT': VT}

    return params

def _calculate_FA_projections(Y, *, C, d, R, 
                              U=None, 
                              D=None, 
                              VT=None,
                              orthonormal=True,
                              **kwargs):
    # Set latent dimensionality
    p = C.shape[1]

    # Calculate projections of X | Y
    R_inv = np.diag(1.0 / R[np.arange(len(R)), np.arange(len(R))])
    B = np.eye(p) + C.T.dot(R_inv.dot(C)) # p x p
    A_inv = R_inv - R_inv.dot(C.dot(np.linalg.inv(B).dot(C.T.dot(R_inv)))) # matrix inversion lemma
    X = np.matmul(C.T, np.matmul(A_inv, (Y - d)))

    # Linearly tranform x into othornomal space U
    if orthonormal:
        if (D is not None) and (VT is not None):
            X = np.matmul(D, np.matmul(VT, X))
        else:
            raise ValueError('Parameters for orthonormal transform (D, VT)'
                             ' must be provided.')
    
    return X

def _calculate_backprojections(X, C, d):
    """
    NOTE: If X was transformed into orthonormal space, then U must be
    provided in place of C.
    """
    return np.matmul(C, X) + d

def calculate_error(Y,
                    method='PCA',
                    err_type='bp',
                    **kwargs):
    """
    Because PCA parameters can be computed once and reused for all number
    of latent dimensions, separate error functions by PCA, and PPCA/FA.
    """
    
    if method.lower() == 'pca':
        return _PCA_error(Y, err_type, **kwargs)
    elif method.lower() in ['ppca', 'fa']:
        return _PPCA_FA_error(Y, err_type, method, **kwargs)
    else:
        raise SyntaxError('Unknown method \'%s\'.' % method)

def _PCA_error(Y,
               err_type,
               p_range=None,
               verbose=False):
    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins
    
    # Setup
    if p_range is None:
        p_range = np.arange(1, q+1)
    error = np.zeros(len(p_range))

    # Iterate over all trials
    for n in range(N):
        if verbose:
            print('Calculating error for trial %d of %d...' % (n+1, N))
            it = tqdm_notebook(p_range)
        else:
            it = p_range

        # Separate train (-n) and test (n) data
        Y_train = np.delete(Y, n, axis=0) # remove nth trial
        Y_train = Y_train.transpose([1, 2, 0]).reshape([q, -1], order='F')
        Y_test = Y[n].reshape([q, -1], order='F')

        # Calculate PCA parameters
        params = _calculate_params(Y_train, method='PCA', p=q)
        
        # Calculate predictions for dimensionality p
        for i, p_i in enumerate(it):
            # Set parameter space p_i
            U_p = params['U'][:, :p_i]
            d = params['d']

            # Back-projection error
            if err_type.lower() == 'bp':
                # Calculate back-projection on test trial
                Y_test_hat = U_p.dot(U_p.T.dot(Y_test - d)) + d

                # Calculate error
                error[i] += np.sum((Y_test_hat - Y_test)**2)

            # Leave-out-neuron prediction error
            elif err_type.lower() == 'loocv':
                # Vectorized form
                err = (np.eye(q) - U_p.dot(U_p.T) + np.eye(q)*(U_p.dot(U_p.T))).dot(Y_test - d)
                error[i] += np.sum(err**2)
                
                # Elementwise form
                # Calculate prediction for neuron j
                #for j in range(q):
                    # Calculate PC scores with jth neuron left out
                    #idx = np.ones(q, dtype=np.bool)
                    #idx[j] = False
                    #X_hat = U_p[idx, :].T.dot(Y_test[idx, :] - d[idx, :]) 

                    # Calculate back-projection from PC scores
                    #Y_test_hat = U_p.dot(X_hat) + d

                    # Calculate error
                    #error[i] += np.sum((Y_test_hat[j] - Y_test[j])**2)

    # Normalize error
    error = (1/(Y.size)) * (error**0.5)

    return error

def _PPCA_FA_error(Y,
                   err_type,
                   method,
                   p_range=None,
                   verbose=False):
    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins
    
    # Setup
    if p_range is None:
        p_range = np.arange(1, q+1)
    error = np.zeros(len(p_range))

    # Iterate over all trials
    for n in range(N):
        if verbose:
            print('Calculating error for trial %d of %d...' % (n+1, N))
            it = tqdm_notebook(p_range)
        else:
            it = p_range

        # Separate train (-n) and test (n) data
        Y_train = np.delete(Y, n, axis=0) # remove nth trial
        #Y_train = Y_train.transpose([1, 2, 0]).reshape([q, -1], order='F')
        Y_test = Y[n].reshape([q, -1], order='F')
        
        # Calculate predictions for dimensionality p
        for i, p_i in enumerate(it):
            # Calculate parameters and projections for latent space
            params, X = _apply_two_step_method(Y_train, method, p=p_i, orthonormal=False)

            # Back-projection error
            if err_type.lower() == 'bp':
                # Calculate back-projection on test trial
                Y_test_hat = _calculate_backprojections(X, params['C'], params['d'])

                # Calculate error
                error[i] += np.sum((Y_test_hat - Y_test)**2)
            
            # Leave-out-neuron prediction error
            elif err_type.lower() == 'loocv':
                # Parameters placeholder
                params_i = {k:v for k, v in params.items()}

                # Calculate prediction for neuron j
                for j in range(q):
                    # Set input with jth neuron left out
                    idx = np.ones(q, dtype=np.bool)
                    idx[j] = False
                    params_i = {'C': params['C'][idx, :],
                                'd': params['d'][idx, :]}

                    # Calculate PC scores with jth neuron left out
                    X_test_hat = _calculate_projections(Y_test[idx, :], method, **params_i)

                    # Calculate back-projection from PC scores
                    Y_test_hat = _calculate_backprojections(X_test_hat, params['C'], params['d'])

                    # Calculate error
                    error[i] += np.sum((Y_test_hat[j] - Y_test[j])**2)

    # Normalize error
    error = (1/(Y.size)) * (error**0.5)

    return error

# TODO: fix need for global precomputed matrices
# implement class?
PRECOMP_1 = None
PRECOMP_2 = None
def _calculate_GPFA_params(Y,
                           dt_bin=1.0,
                           **kwargs):
    gpfa = _GPFA(Y, dt_bin=dt_bin)
    return gpfa.fit(**kwargs)

def _calculate_GPFA_projections(Y, *, C, d, R, K, 
                                U=None, 
                                D=None, 
                                VT=None,
                                orthonormal=True,
                                **kwargs):
    # Shapes
    N = Y.shape[0] # number of trials
    q = Y.shape[1] # number of units
    T = Y.shape[2] # number of time bins

    # Set latent dimensionality
    p = C.shape[1]

    # Find expectation for each trial
    E_xt = np.zeros([N, p, T])
    for n in range(N):
        E_xt[n], _, _ = _gpfa_expectation(Y[n], C, d, R, K)

    # Linearly tranform x into othornomal space U
    if orthonormal:
        if (D is not None) and (VT is not None):
            E_xt_o = np.zeros(E_xt.shape)
            for n in range(N):
                E_xt_o[n] = D.dot(VT.dot(E_xt[n]))
        else:
            raise ValueError('Parameters for orthonormal transform (D, VT)'
                             ' must be provided.')
    
    return E_xt_o

class _GPFA:

    def __init__(self,
                 Y,
                 dt_bin=1.0,
                 **kwargs):


        # Data
        self.Y = Y
        assert Y.ndim == 3
        self.dt_bin = dt_bin

        # Shapes
        self.N = self.Y.shape[0] # number of trials
        self.q = self.Y.shape[1] # number of units
        self.T = self.Y.shape[2] # number of time bins

        # Parameter placeholders
        self.C = None
        self.d = None
        self.R = None
        self.K = None
        self.tau = None

    def fit(self,
            p=1, 
            C_init=None, 
            d_init=None, 
            R_init=None,
            tau_0=1.0,
            sigma_n=1e-3,
            EM_steps=500,
            alpha = 1e-8,
            orthonormal=True, 
            verbose=False):
        # Set latent dimensionality
        self.p = p

        # Shapes
        N = self.N
        q = self.q
        T = self.T

        # Initialize model parameters using FA
        params, _ = _apply_two_step_method(self.Y,
                                           'fa',
                                           p=p, 
                                           C_init=C_init,
                                           d_init=d_init,
                                           R_init=R_init,
                                           EM_steps=EM_steps,
                                           orthonormal=False)
        self.C, self.d, self.R = params['C'], params['d'], params['R']

        # Initialize timescales to t_0
        self.tau = tau_0/self.dt_bin * np.ones(p)

        # Set fixed variance
        sigma_n = (sigma_n * np.ones(p))**0.5
        sigma_f = (1.0 - sigma_n**2)**0.5

        # Precompute matrices
        self.T_1 = np.ones([T, T]) * np.arange(1, T+1)[:, np.newaxis]
        self.T_2 = np.ones([T, T]) * np.arange(1, T+1)

        # Initialize squared exponential covariance matrix K
        self._update_K(sigma_f, sigma_n)
        
        # Placeholders if needed
        #E = []
        #params = []

        # Set progress bar
        if verbose:
            it = tqdm_notebook(range(EM_steps))
        else:
            it = range(EM_steps)
        
        # EM algorithm
        for i in it: 
            # Cache parameters
            #C_old = C
            #d_old = d
            #R_old = R
            #K_old = K
            #tau_old = tau

            ### Optimize FA parameters via EM algorithm ###
            # Get expectations for each trial
            E_xt, E_xtxtT, E_xiTxi = self._expectation(EM_steps)
            
            # Calculate updates over all trials
            self._update_Cd(E_xt, E_xtxtT)
            self._update_R(E_xt)
            
            ### Optimize GP parameters via gradient descent ###
            self._fit_GP_gd(sigma_f, 
                            sigma_n, 
                            E_xiTxi, 
                            alpha=alpha, 
                            max_steps=10000, 
                            epsilon=1e-5)
            #K = np.zeros([T, T, p])
            #for j in range(p):
                #tau[j], K[:, :, j] = _gpfa_fit_GP_auto(tau[j], 
                #                                       sigma_f[j], 
                #                                       sigma_n[j], 
                #                                       E_xiTxi[:, :, :, j], 
                #                                       use_grad=True)

            # Track progress        
            #E.append({'E_xt': E_xt, 'E_xtxtT': E_xtxtT, 'E_xiTxi': E_xiTxi})
            #params.append({'C': C, 'd': d, 'R': R, 'tau': tau})

            # Check for convergence
            #if (np.isclose(C, C_old).all()
            #    and np.isclose(d, d_old).all()
            #    and np.isclose(R, R_old).all()
            #    and np.isclose(K, K_old).all()
            #    and np.isclose(tau, tau_old).all()):
            #    if verbose:
            #        print('Converged after %d iterations.' % (i+1))
            #    break

        # Transform to othornormal space
        if orthonormal:
            U, D, VT = np.linalg.svd(C, full_matrices=False)
            D = np.diag(D)
        else:
            U, D, VT = None, None, None

        # Package params
        params = {'C': C,
                'd': d,
                'R': R,
                'K': K,
                'U': U,
                'D': D,
                'VT': VT,
                'tau': tau,
                'sigma_f': sigma_f,
                'sigma_n': sigma_n}

        return params

    # GPFA EM functions
    def _expectation(self, EM_steps):
        # Shapes
        p = self.p # number of latent dimensions
        N = self.N # number of trials
        q = self.q # number of units
        T = self.T # number of time bins

        # Convert parameters to concatenated and block diagonal forms
        C_bar = np.kron(np.eye(T), self.C)
        R_bar = np.kron(np.eye(T), self.R)
        d_bar = (self.d * np.ones([q, T])).reshape([-1, 1], order='F')
        K_bar = np.zeros([p*T, p*T])
        t = time.time()
        for tt_1 in range(T):
            for tt_2 in range(T):
                K_bar[p*tt_1:p*(tt_1+1), p*tt_2:p*(tt_2+1)] = np.eye(p) * self.K[tt_1, tt_2, :]
        #print('Creating K_bar: %03.3f s' % (time.time()-t))

        # Precompute matrices needed for expectation
        K_bar_inv = np.linalg.inv(K_bar) # TODO: speed this up?
        R_bar_inv = np.kron(np.eye(T), np.linalg.inv(self.R))
        B_inv = np.linalg.inv(K_bar_inv + C_bar.T.dot(R_bar_inv.dot(C_bar)))
        A_inv = R_bar_inv - R_bar_inv.dot(C_bar.dot(B_inv.dot(C_bar.T.dot(R_bar_inv))))
        #A = C_bar.dot(K_bar.dot(C_bar.T)) + R_bar
        #A_inv = np.linalg.inv(A)
        M = K_bar.dot(C_bar.T.dot(A_inv)) # for computing E_x
        Cov_x = K_bar - K_bar.dot(C_bar.T.dot(A_inv.dot(C_bar.dot(K_bar)))) # covariance

        # Placeholders
        E_xt = np.zeros([N, p, T])
        E_xtxtT = np.zeros([N, p, p, T])
        E_xiTxi = np.zeros([N, T, T, p])

        for n in range(N):
            # Reshape data into unrolled vector
            y_bar = self.Y[n].reshape([-1, 1], order='F')
            
            # Compute expectation of neural trajectory
            t = time.time()
            E_x = M.dot(y_bar - d_bar)
            E_xxT = Cov_x + E_x.dot(E_x.T)
            #print('Computed expectation: %03.3f s' % (time.time()-t))
            
            # Slice into three desired expectations for readability
            E_xt[n, :, :] = np.reshape(E_x, [p, T], order='F')
            t = time.time()
            for tt in range(T):
                E_xtxtT[n, :, :, tt] = E_xxT[p*tt:p*(tt+1), p*tt:p*(tt+1)]
            #print('Copied E_xtxtT: %03.3f s' % (time.time()-t))
            t = time.time()
            for j in range(p):
                idx = np.meshgrid(*[np.arange(j, p*T, p)]*2, indexing='ij')
                E_xiTxi[n, :, :, j] = E_xxT[idx]
            #print('Copied E_xtxtT: %03.3f s' % (time.time()-t))
        
        return E_xt, E_xtxtT, E_xiTxi

    def _update_Cd(self, E_xt, E_xtxtT):
        # Shapes
        p = self.p
        N = self.N # number of trials
        q = self.q # number of units
        T = self.T # number of time bins

        # Placeholder
        Cd = np.zeros([q, p+1])

        # Iterate over all trials
        for n in range(N):
            A = np.hstack([self.Y[n].dot(E_xt[n].T), np.sum(self.Y[n], axis=1, keepdims=True)])
            B = np.block([
                [np.sum(E_xtxtT[n], axis=2), np.sum(E_xt[n], axis=1, keepdims=True)],
                [np.sum(E_xt[n], axis=1, keepdims=True).T,  T                  ]
            ])
            Cd += 1/N * A.dot(np.linalg.inv(B))
        
        # Separate C and d
        self.C = Cd[:, :-1]
        self.d = Cd[:, -1][:, np.newaxis]
    
    def _update_R(self, E_xt):
        # Shapes
        N = self.N # number of trials
        q = self.q # number of units
        T = self.T # number of time bins

        # Placeholder
        R = np.zeros([q, q])

        # Iterate overa all trials
        for n in range(N):
            R += 1/(N*T) * np.eye(q) * ( (self.Y[n] - self.d).dot((self.Y[n] - self.d).T) 
                                         - (self.Y[n].dot(E_xt[n].T)).dot(self.C.T) )

        # Set new parameters
        self.R = R

    def _update_K(self, sigma_f, sigma_n):
        # Shapes
        p = self.p
        T = self.T

        # Placeholder
        K = np.zeros([T, T, p])
        for i in range(p):
            logits = -(self.T_1 - self.T_2)**2 / (2 * self.tau[i]**2)
            logits[logits < -100] = -np.inf
            K[:, :, i] = sigma_f[i]**2 * np.exp(logits) + sigma_n[i]**2 * np.eye(T)

        # Update parameters
        self.K = K

    # GPFA GP gradient descent functions
    def _fit_GP_gd(self, sigma_f, sigma_n, E_xiTxi, alpha=1e-5, max_steps=10000, epsilon=1e-5):
        # Shapes
        p = self.p
        N = self.N
        q = self.q
        T = self.T

        # Placeholders
        tau_new = np.zeros(p)

        for j in range(p):
            # Initialize values
            tau_i = self.tau[j]

            for step in range(max_steps):
                # Save old tau
                tau_old_i = tau_i
                
                # Update new K_i
                logits = -(self.T_1 - self.T_2)**2 / (2 * tau_i**2)
                logits[logits < -100] = -np.inf
                K_i = sigma_f[j]**2 * np.exp(logits) + sigma_n[j]**2 * np.eye(T)
                
                # Calculate expensive matrices
                K_inv = np.linalg.inv(K_i)
                
                # Update tau via gradient ascent
                dKdtau = N*(sigma_f[j]**2 * (self.T_1 - self.T_2)**2 / tau_i**3 * np.exp(logits))
                dEdK = np.zeros(K_i.shape)
                for n in range(N):
                    dEdK += 0.5 * (-K_inv + K_inv.dot(E_xiTxi[n, :, :, j].dot(K_inv)))
                dEdtau = np.trace(dEdK.T.dot(dKdtau))
                dEdlog_tau = dEdtau * tau_i # log transformation
                log_tau = np.log(tau_i) + alpha * dEdlog_tau # gradient ascent update
                tau_i = np.exp(log_tau) # inverse log transformation
                
                # Break if step size sufficiently small
                if abs(tau_old_i - tau_i) < epsilon:
                    break
            
            # Save new tau
            tau_new[j] = tau_i

        # Update parameters
        self.tau = tau_new
        self._update_K(sigma_f, sigma_n)

    def _fit_GP_auto(tau, sigma_f, sigma_n, E_xiTxi, use_grad=True):
        logE = lambda log_tau: _neg_logE_XY(log_tau, sigma_f, sigma_n, E_xiTxi, return_grad=use_grad)
        res = minimize(logE, np.log(tau), jac=use_grad)
        tau_max = np.exp(res.x)
        
        return tau_max, _gpfa_update_K(sigma_f, sigma_n, tau_max)

    def _neg_logE_XY(log_tau, sigma_f, sigma_n, E_xiTxi, return_grad=False):
        # Inverse log transformation
        tau_i = np.exp(log_tau)
        
        # Update new K_i
        K_i = update_K(sigma_f, sigma_n, tau_i)
        
        # Calculate expensive matrices
        K_inv = np.linalg.inv(K_i)
        K_det = np.linalg.det(K_i)
        
        # Calculate expected log joint probability
        # that is *only* dependent on terms dependent on tau (i.e. K)
        logE_XY = -0.5*N*T*K_det
        for n in range(N):
            for t in range(T):
                logE_XY += -0.5*np.trace(K_inv.dot(E_xiTxi[n]))
        
        # Calculate gradient
        logits = -(t_1 - t_2)**2 / (2 * tau_i**2)
        logits[logits < -100] = -np.inf
        dKdtau = (sigma_f**2 * (t_1 - t_2)**2 / tau_i**3 * np.exp(logits))*N
        dlogEdK = np.zeros(K_i.shape)
        for n in range(N):
            dlogEdK += 0.5 * (-K_inv + K_inv.dot(E_xiTxi[n].dot(K_inv)))
        dlogEdtau = np.trace(dlogEdK.T.dot(dKdtau))
        dlogEdlog_tau = dlogEdtau * tau_i # log transformation
        
        if return_grad:
            return -logE_XY, -dlogEdlog_tau
        else:
            return -logE_XY