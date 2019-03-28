### Functions for future ephys-analysis repo ###
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from util import _check_list, MDAReader
from mountainlab_pytools.mdaio import writemda64


### MountainSort functions ###
def create_merge_map(*, firings_old, firings_new, merge_map_out=None):
    """
    Maps cluster labels from before to after curation step to track merges.
    """
    # Load firing data if filepaths provided
    if isinstance(firings_old, str):
        firings_old_filepath = firings_old
        with open(firings_old, 'rb') as f:
            firings_old = MDAReader(f).read()
    else:
        firings_old_filepath = ''
    if isinstance(firings_new, str):
        firings_new_filepath = firings_new
        with open(firings_new, 'rb') as f:
            firings_new = MDAReader(f).read()
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
        timeseries = MDAReader(open(timeseries, 'rb')) 
    if isinstance(firings, str):
        with open(firings, 'rb') as f:
            firings = MDAReader(f).read()

    # Set parameters
    cluster_labels = np.unique(firings[2, :])
    num_clusters = len(cluster_labels)
    win_wf = (np.array(window)/1000 * f_s).astype(np.int64) # ms --> samples
    win_len = np.diff(win_wf)[0]
    wfs = np.zeros([num_clusters, timeseries.num_channels, win_len])
    
    # Iterate through all units
    for i, label in enumerate(cluster_labels):
        t = time()
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
        
        print('done. (%.3f seconds)' % (time() - t))
        
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
                templates = MDAReader(f).read()
    
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
            firings = MDAReader(f).read()
    if isinstance(metrics, str):
        with open(metrics, 'r') as f:
            metrics = json.loads(f.read())
    
    # Make lists if needed
    keep_tags = _check_list(keep_tags)
    exclude_tags = _check_list(exclude_tags)

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
        
    
