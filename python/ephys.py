### Functions for future ephys-analysis repo ###
# General imports
import json
import warnings
import time
from tqdm import tqdm_notebook
import operator
from collections import namedtuple

# Plotting tools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Numerical analysis
import math
from scipy.optimize import Bounds, minimize # Poisson
from scipy import stats
import numpy as np
# NOTE: Currently cupy does not implement a mirror of several required numpy
# functions, including np.block (GPFA), np.delete (error), np.linalg.eig (PCA),
# so cupy does not work reliably for functions in this module.
#try:
#    import cupy as np
#    print('cupy successfully imported.')
#except ImportError as e:
#    import numpy as np
#    print('Error importing cupy. Defaulting to numpy.')

# MountainSort
try:
    from mountainlab_pytools.mdaio import writemda64
    from mountainlab_pytools import mlproc as mlp
except ModuleNotFoundError as e:
    print('mountainlab_pytools module not installed. Some functions'
          ' from the ephys package may not be available.')

# Custom modules
import util


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
    'bursting_parents': 'true', # must be string to avoid confusion with 1.0
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
    util.recursive_dict_search(params, 'True', 'true')
    util.recursive_dict_search(params, 'False', 'false')

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

    # Save sort parameters
    with open(output_dir + 'sort_params.json', 'w') as f:
        json.dump(params, f, indent=4)

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
                      ignore_invalid=False,
                      padding_type=None,
                      padding_length=0.0,
                      **kwargs):
    
    # Create kernel if not provided
    if kernel is None:
        kernel = _create_smoothing_kernel(kernel_type, dt_bin ,**kwargs)
        
    # Create padding if desired
    if padding_type is not None:
        if counts.ndim > 1:
            raise SyntaxError('Padding for multidimensional data not currently supported.')
        
        # Offer options similar to MATLAB:
        # https://www.mathworks.com/help/images/ref/imfilter.html#btsmcj2-3
        n_pad = round(padding_length/dt_bin)
        if not isinstance(padding_type, str):
            # Assume scalar provided.
            pad = padding_type*np.ones([2, n_pad])
        elif padding_type == 'symmetric':
            # Mirror edges.
            pad = np.vstack([counts[:n_pad:-1], 
                             counts[-n_pad::-1]])
        elif padding_type == 'replicate':
            # Extend edges.
            pad = np.vstack([np.ones([n_pad])*counts[0],
                             np.ones([n_pad])*counts[-1]])
        elif padding_type == 'circular':
            # Assume circular array.
            pad = np.vstack([counts[-n_pad:],
                             counts[:n_pad]])
        else:
            raise ValueError(f'Unknown padding type \'{padding_type}\'.')

        # Pad array.
        counts = np.hstack([pad[0,:], counts, pad[1,:]])
    else:
        n_pad = 0
    
    # Smooth counts by convolving with kernel
    n_smooth = _convolve(counts, kernel, axis=axis, ignore_invalid=ignore_invalid)
    
    return n_smooth[n_pad:len(n_smooth)-n_pad]
    

def _create_smoothing_kernel(kernel_type, dt_bin, **kwargs):
    if kernel_type == 'Gaussian':
        sigma = kwargs.get('sigma', 0.100) # kernel width (s)
        sigma_k = sigma / dt_bin # kernel width (bins)
        return lambda x: 1/(2*math.pi*sigma_k**2)**0.5 * np.exp(-0.5 * x**2 / sigma_k**2)
    else:
        raise SyntaxError('Unknown kernel type "%s".' % kernel_type)

    
def _convolve(x, k, axis=0, ignore_invalid=False):
    if axis < 0:
        axis = x.ndim + axis
    a = axis
    b = x.ndim - axis - 1
    x_smooth = np.zeros(x.shape)
    for i in range(x.shape[axis]):
        slc = tuple([slice(None)]*a + [i] + [slice(None)]*b)
        k_i = k(np.arange(x.shape[axis]) - i) # zero-center mean
        k_i /= np.sum(k_i) # normalize kernel weights
        k_i = k_i[tuple([np.newaxis]*a + [slice(None)] + [np.newaxis]*b)] # broadcast shape
        if ignore_invalid:
            with np.errstate(invalid='ignore'):
                x_smooth[slc] = np.sum(np.ma.masked_invalid(k_i*x), axis=axis)
        else:
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
class LatentModel:

    def __init__(self, *, p):
        """
        Performs dimensionality reduction on spike counts or firing rates.
        """
        # Set model parameters
        self.p = p

        # Set training data placeholder
        self.Y = None

    def _check_data(self, Y):
        # Initial training data
        if self.Y is None:
            if Y.ndim != 3:
                raise SyntaxError('Training data must have three dimensions.')
            elif (np.sum(Y, axis=(0, 2)) == 0).any():
                warnings.warn('Training data contains a row of zeros. This may '
                              'lead to errors fitting data.', UserWarning, 2)
        # Test data
        elif self.Y.shape[1] not in Y.shape:
            raise SyntaxError('Test data does not have the proper input shape.')

    def fit(self, Y, **kwargs):
        """
        Fits model to provided spike data.

        Args:
        - Y (ndarray, [N, q, T]): Array containing the spike information. Note that
            it must be in the shape [# trials, # units, # time bins]. 
        
        Returns:
        - self
        """
        # Check data format
        self._check_data(Y)

        # Data
        self.Y = Y
        assert Y.ndim == 3

        # Shapes (for reference)
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        return self._fit(**kwargs)

    def _fit(self, **kwargs):
        raise NotImplementedError

    def get_params(self, *params):
        # Return all parameters if none provided
        if len(params) == 0:
            params = self.PARAM_NAMES
        
        param_dict = {}
        for name in params:
            if name in self.PARAM_NAMES:
                param_dict[name] = getattr(self, '_'+name)
            else:
                raise ValueError('Unknown parameter \'%s\'.')
        
        return param_dict

    def set_params(self, **params):
        for k, v in params.items():
            if k in self.PARAM_NAMES:
                setattr(self, '_'+k, v)
            else:
                raise ValueError('Unknown parameter \'%s\'.')
        
        return self

    def project(self, Y, **kwargs):
        # Check data format
        self._check_data(Y)

        return self._project(Y, **kwargs)

    def _project(self, Y, **kwargs):
        raise NotImplementedError

    def backproject(self, Y=None, X=None, **kwargs):
        if X is None:
            X = self.project(Y, **kwargs)
        
        return self._backproject(X, **kwargs)
        
    def _backproject(self, X, **kwargs):
        raise NotImplementedError

    def error(self, Y, err_type, p_range=None, verbose=False, **kwargs):
        # Shapes
        N = Y.shape[0] # number of trials
        q = Y.shape[1] # number of units
        T = Y.shape[2] # number of time bins
        
        # Setup
        if p_range is None:
            p_range = np.arange(1, q)
        error = np.zeros(len(p_range))

        # Iterate over all trials
        for n in range(N):
            # Set iterator
            if verbose:
                print('Calculating error for trial %d of %d...' % (n+1, N))
                it = tqdm_notebook(p_range)
            else:
                it = p_range

            # Separate train (-n) and test (n) data
            Y_train = np.delete(Y, n, axis=0) # remove nth trial
            Y_test = Y[n].reshape([q, -1], order='F')

            # Calculate predictions for dimensionality p
            for i, p_i in enumerate(it):
                # Calculate parameters and projections for latent space
                self.p = p_i
                self.fit(Y_train, **kwargs)

                # Back-projection error
                if err_type.lower() == 'bp':
                    # Calculate back-projection on test trial
                    Y_test_hat = self.backproject(Y_test)

                    # Calculate error
                    error[i] += np.sum((Y_test_hat - Y_test)**2)
                
                # Leave-out-neuron prediction error
                elif err_type.lower() == 'loocv':
                    # Cache parameters
                    cache = {name: np.copy(getattr(self, '_'+name)) for name in self.PARAM_NAMES}

                    # Calculate prediction for neuron j
                    for j in range(q):
                        # Set input with jth neuron left out
                        for name, param in cache.items():
                            dim = np.argwhere(np.array(param.shape) == q).flatten()
                            for k, d in enumerate(dim):
                                param = np.delete(param, j, axis=d)
                            setattr(self, '_'+name, param)

                        # Calculate backprojections with jth neuron left out
                        Y_test_hat = self.backproject(Y_test[idx, :])

                        # Calculate error
                        error[i] += np.sum((Y_test_hat[j] - Y_test[j])**2)

        # Normalize error
        error = (1.0/(Y.size)) * (error**0.5)

        return error

class PCA(LatentModel):

    PARAM_NAMES = ['U', 'lam', 'd']

    def __init__(self, *, p):
        # Initialize base class
        super().__init__(p=p)
        
        # Set parameter placeholders
        for name in self.PARAM_NAMES:
            if ('_' + name) not in self.__dict__:
                setattr(self, '_' + name, None)

    def _fit(self):
        """
        """
        # Shapes
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        # Unroll into single series
        Y_rs = self.Y.transpose([1, 2, 0]).reshape([q, -1], order='F')

        # Calculate eigenvectors of sample covariance matrix
        self._d = np.mean(Y_rs, axis=1, keepdims=True) # sample mean
        S = 1/(N*T) * (Y_rs - self._d).dot((Y_rs - self._d).T) # sample covariance
        lam, U = np.linalg.eig(S) # eigendecomposition of sample covariance
        
        # Sort by eigenvalues
        sort_idx = np.argsort(lam)[::-1]
        self._lam = lam[sort_idx]
        self._U = U[:, sort_idx]

        # Keep first p dimensions
        #self._lam = lam[:self.p]
        #self._U = U[:, :self.p]

        return self

    def _project(self, Y_test, **kwargs):
        """
        """
        # Calculate projections
        X = np.matmul(self._U[:, :self.p].T, (Y_test - self._d)) # preserves dimension order

        return X

    def _backproject(self, X):
        return np.matmul(self._U[:, :self.p], X) + self._d

    def update_p(self, p):
        """
        This avoids having to perform redundant calculations to get different
        number of latent dimensions for the same training data.
        """
        self.p = p

    def error(self, Y, err_type, p_range=None, verbose=False, **kwargs):
        """
        PCA has some features that allows faster error algorithms to be
        implemented, so we will overwrite the generic error function.
        """
        # Shapes
        N = Y.shape[0] # number of trials
        q = Y.shape[1] # number of units
        T = Y.shape[2] # number of time bins
        
        # Setup
        if p_range is None:
            p_range = np.arange(1, q)
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
            Y_test = Y[n].reshape([q, -1], order='F')

            # Calculate PCA parameters
            self.fit(Y_train, **kwargs)
            
            # Calculate predictions for dimensionality p
            for i, p_i in enumerate(it):
                # Set parameter space p_i
                self.p = p_i

                # Back-projection error
                if err_type.lower() == 'bp':
                    # Calculate back-projection on test trial
                    Y_test_hat = self.backproject(Y_test)

                    # Calculate error
                    error[i] += np.sum((Y_test_hat - Y_test)**2)

                # Leave-out-neuron prediction error
                elif err_type.lower() == 'loocv':
                    # Vectorized form
                    U_p = self._U[:, :self.p] # aesthetics
                    err = (np.eye(q) - U_p.dot(U_p.T) + np.eye(q)*(U_p.dot(U_p.T))).dot(Y_test - self._d)
                    error[i] += np.sum(err**2)

        # Normalize error
        error = (1/(Y.size)) * (error**0.5)

        return error

class PPCA(LatentModel):

    PARAM_NAMES = ['C', 'd', 's', 'U', 'D', 'VT']

    def __init__(self, *, p):
        # Initialize base class
        super().__init__(p=p)
        
        # Set parameter placeholders
        for name in self.PARAM_NAMES:
            if ('_' + name) not in self.__dict__:
                setattr(self, '_' + name, None)

    def _fit(self,
             C_init=None, 
             d_init=None, 
             s_init=0.5, 
             EM_steps=500,
             verbose=False,
             **kwargs):
        """
        """
        # Shapes
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        # Unroll into single series
        Y_rs = self.Y.transpose([1, 2, 0]).reshape([q, -1], order='F')

        # Set initial values
        if C_init is None:
            self._C = np.random.rand(q, self.p)
        else:
            self._C = C_init
        if d_init is None:
            self._d = np.mean(Y_rs, axis=1, keepdims=True)
        else:
            self._d = d_init
        self._s = np.array(s_init)

        # Set progress bar
        if verbose:
            it = tqdm_notebook(range(EM_steps))
        else:
            it = range(EM_steps)

        # EM algorithm
        cache_names = ['_' + name for name in self.PARAM_NAMES 
                       if getattr(self, '_' + name) is not None]
        for i in it:
            # Cache parameters
            cache = [np.copy(getattr(self, name)) for name in cache_names]

            # E-step: calculate E(x), E(xx^T) ~ new P(x|y)
            # (which maximizes expected joint probability wrt distribution of y)
            B = self._s * np.eye(self.p) + self._C.T.dot(self._C) # p x p
            A_inv = (1/self._s * np.eye(q)) \
                    - (1/self._s * self._C.dot(np.linalg.inv(B).dot(self._C.T))) # matrix inversion lemma
            E_x = self._C.T.dot(A_inv).dot(Y_rs - self._d) # matrix form
            sum_E_xxT = (np.eye(self.p) - self._C.T.dot(A_inv.dot(self._C)))*(N*T) \
                        + E_x.dot(E_x.T) # summation form
            
            # M-step: update C and R to maximize likelihood
            # (which maximizes expected joint probability wrt parameters)
            self._C = ((Y_rs - self._d).dot(E_x.T)).dot(np.linalg.inv(sum_E_xxT))
            self._s = 1.0/(q*N*T) * np.trace( (Y_rs - self._d).dot((Y_rs - self._d).T) 
                                              - self._C.dot(E_x.dot((Y_rs - self._d).T)) )

            # Check for convergence
            if all([np.isclose(getattr(self, name), attr_old).all() 
                    for name, attr_old in zip(cache_names, cache)]):
                if verbose:
                    print('Converged after %d iterations.' % (i+1))
                break
            
        # Transform to orthonormal space
        self._U, D, self._VT = np.linalg.svd(self._C, full_matrices=False)
        self._D = np.diag(D)

        return self
        
    def _project(self, Y_test, orthonormal=True, **kwargs):
        """
        """
        # Shapes
        q = self._C.shape[0]

        # Calculate projections of X | Y
        # NOTE: matmul must be used instead of dot because Y.ndim > 2.
        # Additionally, shape is [N, q, T] because matmul broadcasts
        # over matrices residing in last two dimensions.
        B = self._s * np.eye(self.p) + self._C.T.dot(self._C) # p x p
        A_inv = (1.0/self._s * np.eye(q)) \
                - (1.0/self._s * self._C.dot(np.linalg.inv(B).dot(self._C.T))) # matrix inversion lemma
        X = np.matmul(self._C.T, np.matmul(A_inv, (Y_test - self._d))) # matrix form

        # Linearly tranform x into orthonomal space U
        if orthonormal:
            X = np.matmul(self._D, np.matmul(self._VT, X))
        
        return X

    def _backproject(self, X, orthonormal=True, **kwargs):
        if orthonormal:
            return np.matmul(self._U, X) + self._d
        else:
            return np.matmul(self._C, X) + self._d

    
class FA(LatentModel):

    PARAM_NAMES = ['C', 'd', 'R', 'U', 'D', 'VT']

    def __init__(self, *, p):
        # Initialize base class
        super().__init__(p=p)
        
        # Set parameter placeholders
        for name in self.PARAM_NAMES:
            if ('_' + name) not in self.__dict__:
                setattr(self, '_' + name, None)

    def _fit(self,
             C_init=None, 
             d_init=None, 
             R_init=None, 
             EM_steps=500,
             verbose=False,
             **kwargs):
        """
        """
        # Shapes
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        # Unroll into single series
        Y_rs = self.Y.transpose([1, 2, 0]).reshape([q, -1], order='F')

        # Set initial values
        if C_init is None:
            self._C = np.random.rand(q, self.p)
        else:
            self._C = C_init
        if d_init is None:
            self._d = np.mean(Y_rs, axis=1, keepdims=True)
        else:
            self._d = d_init
        if R_init is None:
            self._R = np.diag(np.random.rand(q))
        else:
            self._R = R_init

        # Set progress bar
        if verbose:
            it = tqdm_notebook(range(EM_steps))
        else:
            it = range(EM_steps)

        # EM algorithm
        cache_names = ['_' + name for name in self.PARAM_NAMES 
                       if getattr(self, '_' + name) is not None]
        for i in it:
            # Cache parameters
            cache = [np.copy(getattr(self, name)) for name in cache_names]

            # E-step: calculate E(x), E(xx^T) ~ new P(x|y)
            # (which maximizes expected joint probability wrt distribution of y)
            R_inv = np.diag(1.0/self._R[np.arange(len(self._R)), np.arange(len(self._R))])
            B = np.eye(self.p) + self._C.T.dot(R_inv.dot(self._C)) # p x p
            A_inv = R_inv - R_inv.dot(self._C.dot(np.linalg.inv(B).dot(self._C.T.dot(R_inv)))) # matrix inversion lemma
            E_x = self._C.T.dot(A_inv).dot(Y_rs - self._d) # matrix form
            sum_E_xxT = ( (np.eye(self.p) - self._C.T.dot(A_inv.dot(self._C)))*(N*T) 
                          + E_x.dot(E_x.T) ) # summation form

            # M-step: update C and R to maximize likelihood
            # (which maximizes expected joint probability wrt parameters)
            self._C = ((Y_rs - self._d).dot(E_x.T)).dot(np.linalg.inv(sum_E_xxT))
            self._R = 1.0/(N*T) * np.eye(q) * ( (Y_rs - self._d).dot((Y_rs - self._d).T) 
                                                - self._C.dot(E_x.dot((Y_rs - self._d).T)) )

            # Check for convergence
            if all([np.isclose(getattr(self, name), attr_old).all() 
                    for name, attr_old in zip(cache_names, cache)]):
                if verbose:
                    print('Converged after %d iterations.' % (i+1))
                break
            
        # Transform to orthonormal space
        self._U, D, self._VT = np.linalg.svd(self._C, full_matrices=False)
        self._D = np.diag(D)

        return self
        
    def _project(self, Y_test, orthonormal=True, **kwargs):
        """
        """
        # Shapes
        q = self._C.shape[0]

        # Calculate projections of X | Y
        R_inv = np.diag(1.0/self._R[np.arange(len(self._R)), np.arange(len(self._R))])
        B = np.eye(self.p) + self._C.T.dot(R_inv.dot(self._C)) # p x p
        A_inv = R_inv - R_inv.dot(self._C.dot(np.linalg.inv(B).dot(self._C.T.dot(R_inv)))) # matrix inversion lemma
        X = np.matmul(self._C.T, np.matmul(A_inv, (Y_test - self._d)))

        # Linearly tranform x into othornomal space U
        if orthonormal:
            X = np.matmul(self._D, np.matmul(self._VT, X))
        
        return X

    def _backproject(self, X, orthnormal=True, **kwargs):
        if orthonormal:
            return np.matmul(self._U, X) + self._d
        else:
            return np.matmul(self._C, X) + self._d

class GPFA(LatentModel):

    PARAM_NAMES = ['C', 'd', 'R', 'K', 'sigma_f', 'sigma_n', 'tau', 'U', 'D', 'VT']

    def __init__(self, *,
                 p,
                 sigma_n=1e-3):
        # Initialize base class
        super().__init__(p=p)

        # Initialize data-independent GP parameters
        self._sigma_n = (sigma_n * np.ones(self.p))**0.5 # variance
        self._sigma_f = (1.0 - self._sigma_n**2)**0.5
        
        # Set parameter placeholders
        for name in self.PARAM_NAMES:
            if ('_' + name) not in self.__dict__:
                setattr(self, '_' + name, None)
        

    def _fit(self,
             dt_bin=1.0,
             C_init=None, 
             d_init=None, 
             R_init=None,
             tau_0=1.0,
             EM_steps=500,
             alpha = 1e-8,
             verbose=False):

        # Shapes
        N = self.Y.shape[0]
        q = self.Y.shape[1]
        T = self.Y.shape[2]

        # Initialize FA parameters
        if verbose:
            print('Fitting initial FA parameters...')
        fa = FA(p=self.p)
        fa.fit(self.Y,
               C_init=C_init,
               d_init=d_init,
               R_init=R_init,
               EM_steps=EM_steps,
               verbose=verbose)
        params = fa.get_params('C', 'd', 'R')
        self._C, self._d, self._R = params['C'], params['d'], params['R']
        fa = None # release memory

        # Precompute matrices for GP updates
        self._T_1 = np.ones([T, T]) * np.arange(1, T+1)[:, np.newaxis]
        self._T_2 = np.ones([T, T]) * np.arange(1, T+1)

        # Initialize data-dependent GP parameters
        self._tau = tau_0/dt_bin * np.ones(self.p) # timescales
        self._update_K() # squared exponential covariance matrix

        # Set progress bar
        if verbose:
            it = tqdm_notebook(range(EM_steps))
            print('Running joint GPFA EM algorithm...')
        else:
            it = range(EM_steps)
        
        # EM algorithm
        cache_names = ['_' + name for name in self.PARAM_NAMES 
                       if getattr(self, '_' + name) is not None]
        for i in it: 
            # Cache parameters
            cache = [np.copy(getattr(self, name)) for name in cache_names]

            # Optimize FA parameters via EM algorithm
            E_xt, E_xtxtT, E_xiTxi = self._expectation(self.Y)
            self._update_Cd(E_xt, E_xtxtT)
            self._update_R(E_xt)
            
            # Optimize GP parameters via gradient descent
            self._fit_GP_gd(E_xiTxi, 
                            alpha=alpha, 
                            max_steps=10000, 
                            epsilon=1e-5)

            # Check for convergence
            if all([np.isclose(getattr(self, name), attr_old).all() 
                    for name, attr_old in zip(cache_names, cache)]):
                if verbose:
                    print('Converged after %d iterations.' % (i+1))
                break

        # Transform to orthonormal space
        self._U, D, self._VT = np.linalg.svd(self._C, full_matrices=False)
        self._D = np.diag(D)

        return self

    # GPFA EM functions
    def _expectation(self, Y):
        # Shapes
        p = self.p # number of latent dimensions
        N = Y.shape[0] # number of trials
        q = Y.shape[1] # number of units
        T = Y.shape[2] # number of time bins

        # Convert parameters to concatenated and block diagonal forms
        C_bar = np.kron(np.eye(T), self._C)
        R_bar = np.kron(np.eye(T), self._R)
        d_bar = (self._d * np.ones([q, T])).reshape([-1, 1], order='F')
        K_bar = np.zeros([p*T, p*T])
        for tt_1 in range(T):
            for tt_2 in range(T):
                K_bar[p*tt_1:p*(tt_1+1), p*tt_2:p*(tt_2+1)] = np.eye(p) * self._K[tt_1, tt_2, :]

        # Precompute matrices needed for expectation
        K_bar_inv = np.linalg.inv(K_bar) # TODO: speed this up?
        R_bar_inv = np.kron(np.eye(T), np.linalg.inv(self._R))
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
            y_bar = Y[n].reshape([-1, 1], order='F')
            
            # Compute expectation of neural trajectory
            E_x = M.dot(y_bar - d_bar)
            E_xxT = Cov_x + E_x.dot(E_x.T)
            
            # Slice into three desired expectations for readability
            E_xt[n, :, :] = np.reshape(E_x, [p, T], order='F')
            for tt in range(T):
                E_xtxtT[n, :, :, tt] = E_xxT[p*tt:p*(tt+1), p*tt:p*(tt+1)]
            for j in range(p):
                idx = np.meshgrid(*[np.arange(j, p*T, p)]*2, indexing='ij')
                E_xiTxi[n, :, :, j] = E_xxT[idx]
        
        return E_xt, E_xtxtT, E_xiTxi

    def _update_Cd(self, E_xt, E_xtxtT):
        # Shapes
        p = self.p # number of latent dimensions
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        # Placeholder
        self._C, self._d = None, None # release memory
        Cd = np.zeros([q, p+1])

        # Iterate over all trials
        for n in range(N):
            A = np.hstack([self.Y[n].dot(E_xt[n].T), np.sum(self.Y[n], axis=1, keepdims=True)])
            B = np.block([
                [np.sum(E_xtxtT[n], axis=2), np.sum(E_xt[n], axis=1, keepdims=True)],
                [np.sum(E_xt[n], axis=1, keepdims=True).T,  T                  ]
            ])
            Cd += 1/N * A.dot(np.linalg.inv(B))
        
        # Set new parameters
        self._C = Cd[:, :-1]
        self._d = Cd[:, -1][:, np.newaxis]
    
    def _update_R(self, E_xt):
        # Shapes
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        # Placeholder
        self._R = None # release memory
        R = np.zeros([q, q])

        # Iterate overa all trials
        for n in range(N):
            R += 1/(N*T) * np.eye(q) * ( (self.Y[n] - self._d).dot((self.Y[n] - self._d).T) 
                                         - (self.Y[n].dot(E_xt[n].T)).dot(self._C.T) )

        # Set new parameters
        self._R = R

    def _update_K(self):
        # Shapes
        p = self.p
        T = self.Y.shape[2]

        # Placeholder
        self._K = None # release memory
        K = np.zeros([T, T, p])

        # Calculate square exponential covariance for each latent dimension
        for i in range(p):
            logits = -(self._T_1 - self._T_2)**2 / (2 * self._tau[i]**2)
            logits[logits < -100] = -np.inf
            K[:, :, i] = self._sigma_f[i]**2 * np.exp(logits) + self._sigma_n[i]**2 * np.eye(T)

        # Update parameters
        self._K = K

    # GPFA GP gradient descent functions
    def _fit_GP_gd(self, E_xiTxi, alpha=1e-5, max_steps=10000, epsilon=1e-5):
        # Shapes
        p = self.p # number of latent dimensions
        N = self.Y.shape[0] # number of trials
        q = self.Y.shape[1] # number of units
        T = self.Y.shape[2] # number of time bins

        # Placeholders
        tau_new = np.zeros(p)

        for j in range(p):
            # Initialize values
            tau_i = self._tau[j]

            for step in range(max_steps):
                # Save old tau
                tau_old_i = tau_i
                
                # Update new K_i
                logits = -(self._T_1 - self._T_2)**2 / (2 * tau_i**2)
                logits[logits < -100] = -np.inf
                K_i = self._sigma_f[j]**2 * np.exp(logits) + self._sigma_n[j]**2 * np.eye(T)
                
                # Calculate expensive matrices
                K_inv = np.linalg.inv(K_i)
                
                # Update tau via gradient ascent
                dKdtau = N*(self._sigma_f[j]**2 * (self._T_1 - self._T_2)**2 / tau_i**3 * np.exp(logits))
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
        self._tau = tau_new
        self._update_K()

    # TODO: Incoporate auto-fit (_fit_GP_auto, _neg_logE_XY) into class
    def _fit_GP_auto(self, sigma_f, sigma_n, E_xiTxi, use_grad=True):
        logE = lambda log_tau: _neg_logE_XY(log_tau, sigma_f, sigma_n, E_xiTxi, return_grad=use_grad)
        res = minimize(logE, np.log(tau), jac=use_grad)
        tau_max = np.exp(res.x)
        
        return tau_max, _gpfa_update_K(sigma_f, sigma_n, tau_max)

    @staticmethod
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

    def _project(self, Y, orthonormal=True):
        """
        """
        # Assert correct shape
        assert Y.ndim == 3

        # Calculate projections of X | Y for each trial
        E_xt, _, _ = self._expectation(Y)

        # Linearly tranform x into othornomal space U
        if orthonormal:
            E_xt = np.matmul(self._D, np.matmul(self._VT, E_xt))
        
        return E_xt

    def _backproject(self, X, orthnormal=True):
        if orthonormal:
            return np.matmul(self._U, X) + self._d
        else:
            return np.matmul(self._C, X) + self._d


### Distributions ###
class Distribution:
    
    def __init__(self):
        self._TestResult = namedtuple('TestResult', ['statistic', 'pvalue'])
        self._name = 'distribution'
        
    @property
    def params(self):
        return self._params()
        
    def _params(self):
        raise NotImplementedError
        
    @property
    def name(self):
        return self._name
        
    def fit(self, X, **kwargs):
        return self._fit(X, **kwargs)
    
    def _fit(self, X, **kwargs):
        raise NotImplementedError
        
    def pdf(self, X, **kwargs):
        return self._pdf(X, **kwargs)
        
    def _pdf(self, X, **kwargs):
        raise NotImplementedError
        
    def cdf(self, X, **kwargs):
        return self._cdf(X, **kwargs)
    
    def _cdf(self, X, **kwargs):
        raise NotImplementedError


class Exponential(Distribution):
    
    def __init__(self):
        super().__init__()
        self._name = 'exponential'
        
        # Set parameters
        self._lam = None
        
    def _params(self):
        return {'lam': self._lam}
    
    def _fit(self, x):
        self._lam = 1.0/np.mean(x)
        
    def _pdf(self, x):
        return self._lam*np.exp(-self._lam*x)
    
    def _cdf(self, x):
        return 1.0 - np.exp(-self._lam*x)
    
    def _inv_cdf(self, F):
        return -(1.0/self._lam)*np.log(1.0 - F)
    
    def goodness_of_fit(self, x):
        # Sort random values
        x = np.sort(x)
        
        # Calculate ecdf
        ecdf = np.arange(len(x))/float(len(x))

        # Get KS statistic
        #return np.max(np.abs(self.cdf(x) - ecdf))
        result = stats.kstest(rvs=x, cdf=self.cdf)
        
        return self._TestResult(result[0], result[1])


class Poisson(Distribution):
    
    def __init__(self, lam, Lam=None, homogeneous=True):
        """
        Creates Poisson distribution model.

        Args:
        - lam: The rate parameter that characterizes a Poisson distribution.
            If creating a homogeneous model, lam is a constant. Otherwise,
            lam is a function that takes time as input and returns lambda.
        - Lam: The integral of the rate parameter function for a nonhomogeneous
            Poisson process. Lam should be a function that takes two inputs
            (time t and duration s) and returns the integral of the lambda 
            function from t to t+s.
        - homogeneous: If true, then lam is constant, and Lam is ignored. If
            False, then lam and Lam are expected to be functions as described.
        """
        super().__init__()
        self._name = 'poisson'
        
        self.homogeneous = homogeneous
        self._set_params(lam, Lam)
        
    def _set_params(self, lam, Lam):
        if self.homogeneous:
            self.lam = lambda t: lam
            self.Lam = lambda t, s: lam*s
        elif Lam is not None:
            self.lam = lam # function of (t)
            self.Lam = Lam # function of (t, s)
        else:
            raise ValueError('Lambda(t,s) must be provided for non-homogeneous processes.')
        
    def _params(self):
        if self.homogeneous:
            lam = self.lam(0)
        else:
            lam = self.lam
        return {'lam': lam,
                'Lam': self.Lam}
        
    def _array(self, a):
        """Ensures object is a numpy array"""
        if not isinstance(a, np.ndarray):
            return np.array([a])
        else:
            return a
       
    def _pdf(self, t, t_0=None):
        """
        Returns probability density of next inter-event interval
        at time t_0: f(t) = lam(t_0) e^(-lam(t_0)*t). Note that this function will be
        the same for all t for a homogeneous process, but will vary with t_0 for a 
        non-homogeneous process.

        Args:
        - t: Duration of the inter-event interval
        - t_0: Current time in process.
        """
        return self.lam(t_0)*np.exp(-self.lam(t_0)*t)
    
    def _cdf(self, t, t_0=None):
        """
        Returns cumulative probability distribution of next inter-event interval
        at time t_0: F(t) = 1 - e^(-lam(t_0)*t). Note that this function will be
        the same for all t for a homogeneous process, but will vary with t_0 for a 
        non-homogeneous process.

        Args:
        - t: Duration of the inter-event interval
        - t_0: Current time in process.
        """
        return 1.0 - np.exp(-self.lam(t_0)*t)
    
    def _inv_cdf(self, F, t_0=None):
        """
        Returns inverse cumulative probability distribution of next inter-event interval
        at time t_0: t(F) = -(1/lam(t_0)) log (1 - F). Note that this function will be
        the same for all F for a homogeneous process, but will vary with t_0 for a 
        non-homogeneous process.
        """
        return -(1.0/self.lam(t_0))*np.log(1.0 - F)
    
    def _factorial(self, n):
        """Returns n!"""
        if isinstance(n, np.ndarray):
            fact = np.zeros(n.size)
            for i, n_i in enumerate(n):
                fact[i] = np.prod(np.arange(max(n_i, 1))+1)
        else:
            fact = np.prod(np.arange(max(n_i, 1))+1)
    
        return fact
    
    def P(self, n, s, t=0):
        """Returns probability that N(s) = n""" 
        Lam = self.Lam(t, s)
        return np.exp(-Lam)*(np.power(Lam, n))/self._factorial(n)
    
    def times(self, interevent=True, **kwargs):
        """Returns event times sampled from Poisson distribution.
        
        Args:
        - interevent: If True, return times between events. 
            If False, return event times.
        - **kwargs: See _interevent_times() and _event_times()
        """
        if interevent:
            return self._interevent_times(**kwargs)
        else:
            return self._event_times(**kwargs)
        
    def _interevent_times(self, n=1, t_0=None, t_max=np.inf, **kwargs):
        """
        Returns interevent times.

        Args:
        - n: Number of interevent times to generate.
        - t_0: Current time in process.
        - t_max: Maximum interval between events to allow.

        Returns:
        - t: Interevent times of size n.
        """
        if (self.homogeneous) or (t_0 is not None):
            F = np.random.uniform(size=n)
            t = self._inv_cdf(F, t_0)
            t[t > t_max] = t_max
            return t
        else:
            t = self._event_times(**kwargs)
            return np.diff(t)
            
    def _event_times(self, s, t=0, t_max=np.inf):
        """
        Returns event times in interval [t, s)

        Args:
        - s: Duration of current process to sample events.
        - t: Start time in current process to sample events.
        - t_max: Maximum interval between events to allow.
        
        Returns:
        - t_event: Time of events in interval [t, s).
        """
        # Convert to arrays
        t = self._array(t)
        s = self._array(s)
        
        if self.homogeneous:
            # No pruning if homogeneous process
            N_s = self._events(s)
            t_event = np.sort(np.random.uniform(low=t, high=t+s, size=N_s))
            return t_event
        else:
            # Get event times with rate lambda_max
            t_0, lam_max = self._lam_max(t=t, s=s, return_argmax=True)
            N_s = self._events(t=t, s=s, t_0=t_0, t_max=t_max)
            t_event = np.sort(np.random.uniform(low=t, high=t+s, size=N_s))
            
            # Prune drip times to generate inhomogeneous process
            lam_t = self.lam(t_event)
            U = np.random.uniform(size=N_s)
            return t_event[U <= (lam_t / lam_max)]
           
    def _lam_max(self, s, t=0, return_argmax=False):
        """
        Returns the maximum rate parameter in interval [t, s).
        """
        neg_lam = lambda t: -self.lam(t)
        bounds = Bounds(t, t+s)
        soln = minimize(neg_lam, np.ones(t.shape), bounds=bounds)
        if return_argmax:
            return soln.x, -soln.fun
        else:
            return -soln.fun
          
    def _events(self, s, t=0, t_0=None, t_max=np.inf):
        """
        Returns number of events N(s) in interval [t, s) sampled from a
        Poisson distribution by generating inter-event intervals.

        Args:
        - s: Duration of current process to sample events.
        - t: Start time in current process to sample events.
        - t_0: Time in current process to calculate lambda
        - t_max: Maximum interval between events to allow. 

        Returns: 
        - n: Number of events in interval [t, s).
        """
        if (not self.homogeneous) and (t_0 is None):
            raise ValueError('t_0 must be specified by non-homogeneous process.')

        # Guess number of events as Poisson mean
        mean = self.Lam(t, s)
        chunk = max(int(mean), 1)
        T = np.cumsum(self.times(interevent=True, n=chunk, t_0=t_0, t_max=t_max))
        idx = np.searchsorted(T, s, side='left')
        n = idx

        # Continue sampling time until time s reached
        while (idx == T.size):
            T = np.cumsum(self.times(interevent=True, n=chunk, t_0=t_0, t_max=t_max)) + T[-1]
            idx = np.searchsorted(T, s, side='left')
            n += idx
            
        return n
    
    def events(self, s, t=0, t_max=np.inf):
        """
        Returns number of events N(s) in interval [t, s) sampled from a
        Poisson distribution by generating inter-event intervals.

        Args:
        - s: Duration of current process to sample events.
        - t: Start time in current process to sample events.
        - t_max: Maximum interval between events to allow. 

        Returns: 
        - n: Number of events in interval [t, s).
        """
        if self.homogeneous:
            return self._events(s, t_max=t_max)
        
        else:
            return len(self._event_times(t=t, s=s, t_max=t_max))
        
    def _fit(self, T, interevent=True):
        """
        Fit Poisson model to distribution of (inter)event times:
            lam_ML = mean(T)
            
        Note that this can only be fit to a homogeneous distribution.
            
        Args:
        - T: Sample set of (inter)event times to fit.
        - intervent: If True, T represents interevent times. If False, T
            represents sequential event times.
        """
        # Get interevent times
        if not interevent:
            T = np.diff(T)
        
        # The maximum-likelihood estimate of a Poisson is the mean count
        # per unit interval. In other words, the mean rate.
        self.homogeneous = True
        self._set_params(1.0/np.mean(T), None)
        
    def goodness_of_fit(self, T, t_0=0, interevent=True):
        """
        Return results of KS test for fitting Poisson distribution to sample
        set of (inter)event times T. Note that the tests will depend on whether
        interevent is specified. If interevent times, then the KS test will use
        the cdf of an exponential decay function as the model. If event times,
        then a uniform distribution will be used.
        """
        # Rescale times (in case process is inhomogeneous)
        if interevent:
            T = np.cumsum(T)
        T = self.Lam(t_0, T-t_0)
        
        if interevent:
            # Sort interevent times
            T = np.sort(np.diff(T))
            
            # Compare cdf and ecdf. Since times were rescaled, cdf now draws
            # from Poisson with unit rate.
            cdf = lambda t: 1.0 - np.exp(-t)
            ecdf = np.arange(len(T))/float(len(T))
            
            # Get KS statistic
            #return np.max(np.abs(cdf - ecdf))
            result = stats.kstest(rvs=T, cdf=cdf)
        
        else:
            # Compare cdf and ecdf. Since times were rescaled, cdf now draws
            # from uniform distribution in [t_0, max(T))
            cdf = lambda t: t
            ecdf = np.arange(len(T))/float(len(T))
            
            # Get KS statistic
            #return np.max(np.abs(cdf - ecdf))
            result = stats.kstest(rvs=T, cdf=cdf)
            
        return self._TestResult(result[0], result[1])


class Gaussian(Distribution):
    
    def __init__(self):
        """
        Creates Gaussian distribution. The class can support multivariate
        distributions, which are automatically handled from the data passed
        to the fit() method.
        """
        super().__init__()
        self._name = 'gaussian'
        
        # Set parameters
        self._mu = None
        self._Sigma = None
        
        # Feature placeholders
        self._m = None # number of dimensions
        
        # Cache costly calculations
        self._den = None # denominator of Gaussian (includes determinant)
        self._Sigma_inv = None # inverse of covariance
        
    def _params(self):
        if self._m == 1:
            return {'mu': self._mu.squeeze()[()],
                    'sigma': self._Sigma.squeeze()[()]**0.5}
        else:
            return {'mu': self._mu,
                    'Sigma': self._Sigma}
    
    def _format_data(self, X):
        if X.ndim == 1 and self._m is None:
            return X[:, np.newaxis]
        elif X.ndim == 2 and self._m is None:
            return X
        elif X.ndim == 1 and X.shape[0] == self._m:
            return X[np.newaxis, :]
        elif X.ndim == 1 and self._m == 1:
            return X[:, np.newaxis]
        elif X.ndim == 2 and X.shape[0] == self._m:
            return X.T
        elif X.ndim == 2 and X.shape[1] == self._m:
            return X
        else:
            raise SyntaxError('Data shape of [' + ','.join([str(n) for n in X.shape]) + '] not understood.')
    
    def _fit(self, X):
        
        X = self._format_data(X)
        
        N = X.shape[0] # number of samples
        m = X.shape[1] # number of dimensions
        
        # Maximum-likelihood estimation
        self._mu = np.mean(X, axis=0)
        self._Sigma = (1.0/N) * (X - self._mu).T.dot(X - self._mu)
        
        # Remember features
        self._m = m
        self._N = N
        
        # Reset cache
        self._den = None
        
    def _pdf(self, X):
        
        X = self._format_data(X)
        
        N = X.shape[0] # number of samples
        m = X.shape[1] # number of dimensions
        
        if self._den == None:
            self._den = np.sqrt((2.0*np.pi)**m * np.linalg.det(self._Sigma))
        if self._Sigma_inv is None:
            self._Sigma_inv = np.linalg.inv(self._Sigma)
        num = np.exp(-0.5*np.diag((X - self._mu).dot(self._Sigma_inv.dot((X - self._mu).T))))
        
        return num / self._den
    
    def _cdf(self, X):
        if self._m == 1:
            # Assume X represents points in 1D space
            X = self._format_data(X)
            return stats.norm.cdf(X, loc=self._mu, scale=np.sqrt(self._Sigma))
        else:
            # Assume X represents Mahalanobis distances from the mean
            assert X.ndim == 1
            return self._multivariate_cdf(self._m, X)
        
    def _multivariate_cdf(self, m, r):
        if m == 1:
            return 2*(stats.norm.cdf(r, loc=0, scale=1) - 0.5)
        elif m == 2:
            return 1.0 - np.exp(-0.5*r**2)
        else:
            return self._multivariate_cdf(m-2, r) - r**(m-2)*np.exp(-0.5*r**2)/(m-2)
    
    def _inv_cdf(self, q):
        if self._m == 1:
            return stats.norm.ppf(q, loc=self._mu, scale=np.sqrt(self._Sigma))
        else:
            raise NotImplementedError('Inverse cdf not yet implemented for multivariate Gaussian.')
    
    def distance(self, X):
        """
        Returns the Mahalanobis distance between sample x and model mean mu:
        
        d^2(x, Sigma) = (x - mu)^T Sigma^-1 (x - mu)
        
        where Sigma and mu are the mean and covariance matrix of the sample set
        used to fit the model.
        
        Args:
        - X: Samples to query distance. Expected format is [N, m], where N is the
            number of samples, and m is the dimensionality.
        """
        # Format data
        X = self._format_data(X)
        
        return self._distance(X)
    
    def _distance(self, X):
        # Compute inverse covariance matrix if not already cached
        if self._Sigma_inv is None:
            self._Sigma_inv = np.linalg.inv(self._Sigma)
        
        # Calculate Mahalanobis distance
        return np.sqrt(np.diag((X - self._mu).dot(self._Sigma_inv.dot((X - self._mu).T))))
    
    def contour(self, t, percentile=0.95):
        """
        Generate the level curve (ellipsoid) corresponding to the confidence interval
        given by percentile. The ellipsoid is given by the equation:

        (x - mu)^T Sigma^-1 (x - mu) = c^2

        where c^2 follows a chi-squared distribution with m degrees of freedom:

        c^2 ~ chi_m^2(percentile)

        For a bivariate (m=2) distribution, the parametric equation is:

        x1' = c/sqrt(lam1) cos(t)
        x2' = c/sqrt(lam2) sin(t)

        where lam1, lam2 are the eigenvalues from the eigendecomposition of Sigma^-1,
        U diag(lam) U^T, and x' = U^T(x - mu) (or in matrix form, X' = (X - mu)U ).

        Args:
        - t: Angle in radians for parametric equation.
        - percentile: Probability that random sample will fall within ellipse.

        Returns:
        - Xp: Matrix of shape [N, m] containing points of the ellipsoid that 
          correspond to t.
        """
        # First, determine value of level set. The Mahalanobis distance of a random
        # sample in R^d from the mean follows a chi-squared distribution with d degrees
        # of freedom.
        c2 = stats.chi2.ppf(percentile, 2)

        # Next, get eigendecomposition of the covariance matrix
        # Note that while the proof generally use the precision matrix
        # (i.e. inverse of covariance matrix), the only difference between 
        # using the two matrices is that eigenvalues are reciprocals. Thus
        # this saves us an extra step of (potentially costly) matrix inversion.
        lam, U = np.linalg.eig(self._Sigma)

        # Get values of elliptical parameters.
        a = (c2*lam[0])**0.5
        b = (c2*lam[1])**0.5

        # Compute elliptical coordinates in transformed system.
        Xp_tf = np.vstack([a*np.cos(t), b*np.sin(t)]).T

        # Transform elliptical coordinates into original system. Note in matrix
        # form this becomes X = X'(U^-1) + mu = X'(U^T) + mu .
        Xp = Xp_tf.dot(U.T) + self._mu
        
        return Xp


    def get_statistic(self, X, distribution='beta'):
        """
        Get the statistic of a set of observations X relative to the fit distribution.
        
        If X is part of the sample set used to fit the distribution, then:
        
        (N d^2)/(N-1)^2 ~ B(m/2, (N-m-1)/2)
        
        where N is the number of samples used to fit the distribution, m is the 
        dimensionality of each sample, and d^2 is the Mahalanobis distance between 
        the observation X and model mean mu.
        
        If X is a new observation independent of the original sample set, then:
        
        (N(N-m)d^2)/(m(N-1)^2) ~ F(m, N-m)
        
        where F is the F-distribution.
        
        (If the distribution was fit to the true population parameters, then chi-squared
        should be used.)
        
        Args:
        - X: Observation set to query. Shape [N' x m].
        - distribution: Type of distribution to test against. Options are beta or f.
        
        Returns:
        - stat: 1D array of length N' containing statistic of each observation in X.
        """
        # Format data
        X = self._format_data(X)
        
        # Use sample set dimensions
        N = self._N
        m = self._m
        
        # Calculate statistic for specified distribution
        if distribution.lower() == 'beta':
            return (N/(N-1)**2) * self._distance(X)**2
        elif distribution.lower() == 'f':
            return (N*(N-m))/(m*(N-1)**2) * self._distance(X)**2
        else:
            raise ValueError('Unknown distribution \'{}\''.format(distribution))
        
    def test_statistic(self, X, distribution='beta', q=0.95, test='>'):
        """
        Test the probability of observation X being a part of the fit distribution.
        See the get_statistic method for an explanation of the different 
        distributions.
        
        Args:
        - X: Observation set to query. Shape [N' x m].
        - distribution: Type of distribution to test against. Options are beta or f.
        - q: Percentile within distribution to test against. In other words, if q = 0.95,
            then the function tests the relationship of observation X to the 95th 
            percentile of the fit distribution.
        - test: Relationship to q to test. Options are >, >=, <, <=.
        
        Returns:
        - result: 1D boolean array of length N' containing results of relationship of 
            each observation to percentile q of distribution.
        """
        # Get statistic
        stat = self.get_statistic(X, distribution=distribution)
        
        # Use sample set dimensions
        N = self._N
        m = self._m
        
        # Calculate statistic at query percentile q
        if distribution.lower() == 'beta':
            stat_q = stats.beta.ppf(q, m/2, (N - m - 1)/2)
        elif distribution.lower() == 'f':
            stat_q = stats.f.ppf(q, m, N - m)
        else:
            raise ValueError('Unknown distribution \'{}\''.format(distribution))    
        
        # Get test operator
        ops = {'<':  operator.lt,
               '<=': operator.le,
               '>':  operator.gt,
               '>=': operator.ge}
        
        return ops[test](stat, stat_q)
    
    def goodness_of_fit(self, X, method='SW'):
        # No multivariate test implemented yet
        if self._m > 1:
            raise SyntaxError('Multivariate test not implemented.')
            
        # Format data
        X = self._format_data(X)
        
        # Kolmogorov-Smirnov test
        if method.lower() == 'ks':
            cdf = lambda x: stats.norm.cdf(x, loc=self._mu, scale=np.sqrt(self._Sigma))
            result = stats.kstest(rvs=X, cdf=cdf)
        # Shaprio-Wilk test
        elif method.lower() == 'sw':
            result = stats.shapiro(X)
        else:
            raise ValueError('Unknown method \'{}\''.format(method))
            
        return self._TestResult(result[0], result[1])
        
    
class LogGaussian(Gaussian):
    
    def __init__(self, **kwargs):
        """
        Creates log-Gaussian distribution, in which the log-transformed
        data is fit to a normal distribution. Note that the log-normal 
        distribution essentially fits the parameters (mean, variance) to
        log-transformed data, but then has a modified pdf due to the 
        logarithmic transformation:

        ln(X) = Y ~ N(mu, sigma**2)
        f(x) = (1/x) f(ln(x))

        that is, the pdf is obtained by treating the log-transformed data
        as a normal distribution, with an additional scaling factor of (1/x).
        For a multivariate distribution, this extends to prod_i(1/x_i). See
        here for more details:
        https://stats.stackexchange.com/a/296843/355903
        https://stats.stackexchange.com/a/297559/355903
        https://stats.stackexchange.com/q/65998/355903


        Because the cdf integrates out these differences, there is no scaling
        factor.
        """
        super().__init__(**kwargs)
        self._name = 'log-gaussian'
        
    def _format_data(self, X):
        # Apply logarithmic transformation first
        X = np.log(X)
        
        if X.ndim == 1 and self._m is None:
            return X[:, np.newaxis]
        elif X.ndim == 2 and self._m is None:
            return X
        elif X.ndim == 1 and X.shape[0] == self._m:
            return X[np.newaxis, :]
        elif X.ndim == 1 and self._m == 1:
            return X[:, np.newaxis]
        elif X.ndim == 2 and X.shape[0] == self._m:
            return X.T
        elif X.ndim == 2 and X.shape[1] == self._m:
            return X
        else:
            raise SyntaxError('Data shape of [' + ','.join([str(n) for n in X.shape]) + '] not understood.')

    def _pdf(self, X):
        """
        Override pdf of normal distribution by applying (1/x) scaling factor.
        """
        # Format data (apply log-transformation)
        X = self._format_data(X)
        
        # Get data shape.
        N = X.shape[0] # number of samples
        m = X.shape[1] # number of dimensions
        
        # Compute denominator and Sigma**-1
        if self._den == None:
            self._den = np.sqrt((2.0*np.pi)**m * np.linalg.det(self._Sigma))
        if self._Sigma_inv is None:
            self._Sigma_inv = np.linalg.inv(self._Sigma)
        num = np.exp(-0.5*np.diag((X - self._mu).dot(self._Sigma_inv.dot((X - self._mu).T))))
        
        # Return pdf, noting that the scaling factor is prod_i(1/x_i).
        return num / (np.prod(np.exp(X), axis=1)*self._den)


### Classification models ###
class Model:
    
    def __init__(self):
        self._name = 'model'
        self._m = None
        
    @property
    def params(self):
        return self._params()
        
    def _params(self):
        raise NotImplementedError
        
    @property
    def name(self):
        return self._name
    
    def _format_data(self, X):
        if X.ndim == 1 and self._m is None:
            return X[:, np.newaxis]
        elif X.ndim == 2 and self._m is None:
            return X
        elif X.ndim == 1 and X.shape[0] == self._m:
            return X[np.newaxis, :]
        elif X.ndim == 1 and self._m == 1:
            return X[:, np.newaxis]
        elif X.ndim == 2 and X.shape[0] == self._m:
            return X.T
        elif X.ndim == 2 and X.shape[1] == self._m:
            return X
        else:
            raise SyntaxError('Data shape of [' + ','.join([str(n) for n in X.shape]) + '] not understood.')
        
    def fit(self, X, *args, **kwargs):
        X = self._format_data(X)
        return self._fit(X, *args, **kwargs)
    
    def _fit(self, X, *args, **kwargs):
        raise NotImplementedError
        
    def predict(self, X, *args, **kwargs):
        X = self._format_data(X)
        return self._predict(X, *args, **kwargs)
        
    def _predict(self, X, *args, **kwargs):
        raise NotImplementedError

    def goodness_of_fit(self, X, *args, **kwargs):
        X = self._format_data(X)
        return self._goodness_of_fit(X, *args, **kwargs)
        
    def _goodness_of_fit(self, X, *args, **kwargs):
        raise NotImplementedError


class LinearRegression(Model):
    PARAM_NAMES = ['w']

    def __init__(self, use_bias=True):
        super().__init__()
        self._name = 'linear-regression'

        # Set parameters
        self._w = None
        self.use_bias = use_bias

    def _params(self):
        return {'w': self._w}

    def _fit(self, X, y):
        # Get parameters
        N = X.shape[0]
        self._m = X.shape[1]

        # Reformat y if needed
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Fit weights using least-squares regression:
        # w = (X^T X )^-1 X^T y
        if self.use_bias:
            X = np.hstack([X, np.ones([N, 1])]) # add bias term
        self._w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    def _predict(self, X):
        N = X.shape[0]
        if self.use_bias:
            X = np.hstack([X, np.ones([N, 1])]) # add bias term
        return X.dot(self._w)

    def _goodness_of_fit(self, X, y, method='r_squared'):
        # Format data
        N = X.shape[0]
        if y.ndim == 1:
            y = y[:, np.newaxis]
        elif y.ndim > 2:
            raise SyntaxError('y must be a 1D or 2D array.')
        
        # Compute components 
        # (see e.g. https://online.stat.psu.edu/stat462/node/95/)
        if method == 'r_squared':
            y_hat = self.predict(X)
            y_mean = np.mean(y, axis=0)
            SSR = np.sum((y_hat - y_mean)**2) # regression sum of squares
            SST = np.sum((y - y_mean)**2) # total sum of squares
            return SSR/SST
        else:
            raise ValueError('Unknown method \'{}\'.'.format(method))


class KMeans(Model):
    
    PARAM_NAMES = ['k', 'c']
    
    def __init__(self, k):
        super().__init__()
        self._name = 'k-means'
        
        # Set parameters
        self._k = k
        self._c = None
        
    def _params(self):
        return {'k': self._k,
                'c': self._c}
    
    def _distance(self, X, c):
        """Calculates Euclidean distance between array X and vector c."""
        # Get shapes
        N = X.shape[0]
        assert X.shape[1] == self._m
        
        # Return L2 distance
        return np.sqrt(np.sum((X - c)**2, axis=1))
        
        
    def _fit(self, X, 
             init_method='data',
             allow_empty=False, 
             max_iters=1000, 
             verbose=True):
        # Get parameters
        N = X.shape[0]
        self._m = X.shape[1]
        
        # Initialize centroids
        if init_method.lower() == 'random':
            # Pick random points within range of data
            c = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=[self._k, self._m])
        elif init_method.lower() == 'data':
            # Pick random points in data
            idx = np.random.choice(np.arange(N), size=self._k, replace=False)
            c = X[idx, :]
        else:
            raise ValueError('Unknown initialization method \'{}\''.format(init_method))
        
        # Run algorithm
        i = 0
        self._c = c
        y = -np.ones([N])
        cache_names = ['_' + name for name in self.PARAM_NAMES 
                       if getattr(self, '_' + name) is not None]
        while(i < max_iters):
            # Cache old parameters
            cache = [np.copy(getattr(self, name)) for name in cache_names]
            
            # Assign points to cluster based on Euclidean distance
            y = self._predict(X)
            
            # Update centroids
            c = np.zeros([self._k, N, self._m])
            c[y, np.arange(N)] = np.copy(X)
            counts = np.zeros([self._k])
            ids, counts_ = np.unique(y, return_counts=True)
            counts[ids] = counts_
            if (ids.size < self._k) and (not allow_empty):
                # Handle orphaned centroids by assigning to farthest point
                for j in np.arange(self._k)[counts == 0]:
                    self._c[j] = X[np.argmax(self._distance(X, self._c[j])), :]
            self._c[counts > 0] = np.sum(c[counts > 0], axis=1)/counts[counts > 0, np.newaxis]
            
            # Check for convergence
            if all([np.isclose(getattr(self, name), attr_old).all() 
                    for name, attr_old in zip(cache_names, cache)]):
                if verbose:
                    print('Converged after %d iterations.' % (i+1))
                break
            
            # Increment counter
            i += 1
    
    def _predict(self, X):
        # Get shape
        N = X.shape[0]
        
        # Assign points to cluster based on Euclidean distance
        dist = np.sum((X[np.newaxis, :, :] - self._c[:, np.newaxis, :])**2, axis=2) # shape=[k, N]
        y = np.argmin(dist, axis=0)
        
        return y 
      
        
class GaussianMixture(Model):
    
    PARAM_NAMES = ['k', 'mu', 'Sigma', 'pi']
    
    def __init__(self, k):
        super().__init__()
        self._name = 'gaussian-mixture'
        
        # Set parameters
        self._k = k
        self._models = None
        self._mu = None
        self._Sigma = None
        self._pi = None
        
    def _params(self):
        return {'k': self._k,
                'mu': self._mu,
                'Sigma': self._Sigma,
                'pi': self._pi}
        
    def _fit(self, X, init_method='kmeans', max_iters=1000, verbose=True):
        # Get parameters
        N = X.shape[0]
        self._m = X.shape[1]
        
        # Initialization:
        self._models = [] # Gaussian models ~ N(mu_k, Sigma_k)
        if init_method == 'kmeans':
            # Via k-means:
            # - Find mu_k via k-means
            # - Find Sigma_k from all points in cluster k
            kmeans = KMeans(self._k)
            kmeans.fit(X, init_method='data', verbose=False)
            labels = kmeans.predict(X) # initial cluster labels
            for label in np.unique(labels):
                idx = np.argwhere(labels == label).flatten()
                model = Gaussian()
                model.fit(X[idx, :])
                self._models.append(model)
        elif init_method == 'random':
            # Via randomization:
            # - Set mu_k as random data point
            # - Set Sigma_k as covariance of all data
            model = Gaussian()
            model.fit(X)
            Sigma = model._Sigma
            idx = np.random.choice(np.arange(N), size=self._k)
            for i in idx:
                model = Gaussian()
                model._m = X.shape[1]
                model._mu = X[i, :]
                model._Sigma = Sigma
                self._models.append(model)
        self._pi = np.ones([self._k])/self._k # uniform priors
            
        # EM algorithm
        i = 0
        cache_names = ['_' + name for name in self.PARAM_NAMES 
                       if getattr(self, '_' + name) is not None]
        while (i < max_iters):
            # Cache old parameters
            cache = [np.copy(getattr(self, name)) for name in cache_names]
            
            # Expectation
            gamma = self._expectation(X)
            
            # Maximization
            mu, Sigma, pi = self._maximization(X, gamma)
                
            # Update parameters
            for k, model in enumerate(self._models):
                model._mu = mu[k]
                model._Sigma = Sigma[k]
            self._mu = mu
            self._Sigma = Sigma
            self._pi = pi
            
            # Check for convergence
            if all([np.isclose(getattr(self, name), attr_old).all() 
                    for name, attr_old in zip(cache_names, cache)]):
                if verbose:
                    print('Converged after %d iterations.' % (i+1))
                break
                
            # Increment counter
            i += 1

        if (i == max_iters) and verbose:
            print('Max iterations reached.' % (i))
            
    def _expectation(self, X):
        # Get shape
        N = X.shape[0]
        
        # Compute probability ratio for each class
        num = np.zeros([N, self._k])
        den = np.zeros([N, 1])
        for k, model in enumerate(self._models):
            num[:, k] = model.pdf(X) * self._pi[k]
            den[:, 0] += model.pdf(X) * self._pi[k]
            
        gamma = num / den
            
        return gamma
    
    def _maximization(self, X, gamma):
        # Get shape
        N = X.shape[0]

        # Compute ML values
        N_k = np.sum(gamma, axis=0)
        mu = (1.0/N_k[:, np.newaxis])*np.sum(gamma[:, :, np.newaxis]*X[:, np.newaxis, :], axis=0)
        Sigma = np.zeros([self._k, self._m, self._m])
        for k in range(self._k):
            Sigma[k] = (1.0/N_k[k])*(X - mu[k]).T.dot(X - mu[k])
        pi = N_k/N
        
        return mu, Sigma, pi
    
    def _predict(self, X):
        # Compute posterior P(k | X)
        p_k = self._expectation(X) # shape=[N, k]
        
        # Assign classes based on maximum a posteriori rule
        return np.argmax(p_k, axis=1).flatten()