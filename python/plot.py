import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from util import get_patch_statistics, _check_list

class Plotter:

    DEFAULT_SETTINGS = {'cmap': 'copper',
                        'style': 'seaborn-deep',
                        'rcParams': {}}

    def __init__(self, **kwargs):
        # Plot settings
        cmap = kwargs.get('cmap', self.DEFAULT_SETTINGS['cmap'])
        self.cmap = plt.get_cmap(cmap)
        self.style = kwargs.get('style', self.DEFAULT_SETTINGS['style'])
        matplotlib.rcParams.update(kwargs.get('rcParams', self.DEFAULT_SETTINGS['rcParams']))
        # TODO: initiate pyplot settings

        # Initial figure
        self.fig = None

    def create_new_figure(self, figsize=(15, 15), rows=1, cols=1):
        # Clear old figure
        plt.close('all')
        
        # Create figure and specified axes
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize)

        # Set current axis to first if multiple axes
        if (rows == 1) and (cols == 1):
            self.axes = np.array([self.axes])
        if (rows == 1) or (cols == 1):
            self.ax = self.axes[0]
        else:
            self.ax = self.axes[0, 0]

        # General plot settings
        for ax in self.axes.flatten():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Axes to show
        self.show = np.ones(self.axes.shape, dtype=np.bool)

    def set_current_axis(self, idx):
        # Numpy arrays support variable number of indices if
        # provided as tuple
        if not isinstance(idx, tuple):
            idx = tuple(idx)
        self.ax = self.axes[idx]

    def plot_learning_curve(self,
                            days,
                            data,
                            day_range=None,
                            center='median',
                            err='sem',
                            label=None,
                            c=0.5,
                            plot_traces=False,
                            plot_points=False):

        def get_plot_idx(d):
            if day_range is None:
                return np.ones(len(d), dtype=np.bool)
            else:
                return np.logical_and(d >= day_range[0], d <= day_range[1])

        # Convert to dictionary if needed (e.g. single animal data)
        if not isinstance(data, dict):
            data = {'mouse': data}
            days = {'mouse': days}

        # Plot individual traces if population data
        n_mouse = len(data.keys())
        for i, mouse_id in enumerate(data.keys()):
            # All sessions
            if plot_traces:
                data_, days_ = get_patch_statistics(data[mouse_id], 
                                                    ids=days[mouse_id], 
                                                    method=center,
                                                    return_all=False)
                plot_idx = get_plot_idx(days_)
                self.ax.plot(days_[plot_idx], 
                             data_[plot_idx], 
                             color=self.cmap((i+1)/n_mouse),
                             marker='o',
                             label=mouse_id)
            if plot_points:
                data_, days_ = get_patch_statistics(data[mouse_id], 
                                                    ids=days[mouse_id], 
                                                    method=center,
                                                    return_all=True)
                plot_idx = get_plot_idx(days_)
                self.ax.scatter(days_[plot_idx], 
                                data_[plot_idx], 
                                color=self.cmap((i+1)/n_mouse),
                                label=mouse_id)
                # not sure what I was doing here...
                #(days_, data_) = (days_[plot_idx], data_[plot_idx])
                #for dt in np.unique(data_): 
                #    idx = (data_ == dt)
                #    n = np.sum(idx)
                #    self.ax.scatter(days_[idx], 
                #                    data_[idx], 
                #                    color=self.cmap((i+1)/n_mouse),
                #                    label=mouse_id)
            
        # Plot overall trace
        data_, days_ = get_patch_statistics(data, 
                                            ids=days, 
                                            method=center, 
                                            return_all=False)
        data_err, days_ = get_patch_statistics(data, 
                                            ids=days, 
                                            method=err, 
                                            return_all=False)
        y_err = np.vstack([np.zeros(len(data_err)), data_err])
        plot_idx = get_plot_idx(days_)
        self.ax.errorbar(days_[plot_idx], 
                         data_[plot_idx], 
                         yerr=y_err[:, plot_idx], 
                         color=self.cmap(c),
                         marker='o',
                         capsize=5,
                         label=label)

    def plot_harvest_rates(self, days, hr, figsize=(10, 10), **kwargs):
        # Create new figure
        self.create_new_figure(figsize=figsize)

        # Heavy lifting
        self.plot_learning_curve(days, hr, **kwargs)

        # Plot settings
        self.ax.set_xlabel('Session')
        self.ax.set_ylabel('Harvest Rate per Patch (uL/s)')
        
    def plot_harvest_diffs(self,
                           days,
                           hr_obs, 
                           hr_opt, 
                           hr_max,
                           figsize=(10, 10),
                           **kwargs):
        # Create new figure
        self.create_new_figure(figsize=figsize)

        # Make dictionaries if needed
        if not isinstance(hr_obs, dict):
            hr_obs = {'mouse': hr_obs}
            hr_opt = {'mouse': hr_opt}
            hr_max = {'mouse': hr_max}
            days   = {'mouse': days}

        # Plot difference between observed and optimal
        hr_diff_opt = {}
        for [mouse_id, hr_opt_mouse], [mouse_id, hr_obs_mouse] in zip(hr_opt.items(), hr_obs.items()):
            hr_diff_opt[mouse_id] = []
            for hr_opt_i, hr_obs_i in zip(hr_opt_mouse, hr_obs_mouse):
                hr_diff_opt[mouse_id] = hr_obs_i - hr_opt_i
        self.plot_learning_curve(days, hr_diff_opt, label='observed vs. optimal', **kwargs)

        # Plot difference between observed and max
        hr_diff_max = {}
        for [mouse_id, hr_max_mouse], [mouse_id, hr_obs_mouse] in zip(hr_max.items(), hr_obs.items()):
            hr_diff_max[mouse_id] = []
            for hr_max_i, hr_obs_i in zip(hr_max_mouse, hr_obs_mouse):
                hr_diff_max[mouse_id] = hr_obs_i - hr_max_i
        self.plot_learning_curve(days, hr_diff_max, label='observed vs. maximum', **kwargs)

        # Plot settings
        self.ax.legend()
        self.ax.set_xlabel('Session')
        self.ax.set_ylabel('Harvest Rate Difference per Patch (uL/s)')

    def plot_residence_times(self, days, t_p_obs, t_p_opt, figsize=(10, 10), **kwargs):
        # Create new figure
        self.create_new_figure(figsize=figsize)

        # Plot observed data
        self.plot_learning_curve(days, t_p_obs, label='observed', **kwargs)

        # Plot optimal data
        center = kwargs.get('center', 'mean')
        day_range = kwargs.get('day_range', None)
        t_p_opt_, days_ = get_patch_statistics(t_p_opt, 
                                               ids=days, 
                                               method=center,
                                               return_all=False)
        if day_range is None:
            plot_idx = np.ones(len(days_), dtype=np.bool)
        else:
            plot_idx = np.logical_and(days_ >= day_range[0], days_ <= day_range[1])
        self.ax.plot(days_[plot_idx], t_p_opt_[plot_idx], 
                linestyle='--', color=self.cmap(0.10), label='optimal')

        # Plot settings
        handles, labels = self.ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        self.ax.legend(handles, labels) # sorted by labels
        self.ax.set_title('Patch Residence Times across Sessions')
        self.ax.set_xlabel('Session')
        self.ax.set_ylabel('Patch Residence Time (s)')
        self.ax.set_yscale('log')
        xlim = self.ax.get_xlim()
        self.ax.set_xlim([xlim[0], xlim[1] + 5])

    def plot_travel_times(self, days, t_t, figsize=(10, 10), **kwargs):
        # Create new figure
        self.create_new_figure(figsize=figsize)

        # Plot observed data
        self.plot_learning_curve(days, t_t, **kwargs)

        # Plot settings
        self.ax.legend()
        self.ax.set_title('Travel Times across Sessions')
        self.ax.set_xlabel('Session')
        self.ax.set_ylabel('Travel Time (s)')
        self.ax.set_yscale('log')
        xlim = self.ax.get_xlim()
        self.ax.set_xlim([xlim[0], xlim[1] + 5])

    def plot_psth(self, *,
                  counts,
                  bins,
                  dt_bin,
                  err=None,
                  labels=None,
                  metrics=None,
                  new_fig=True,
                  **kwargs):
        
        if metrics is not None:
            # Get unit labels if not provided
            if labels is None:
                labels = np.array([c['label'] for c in metrics['clusters']])
        
            # Place label order of metrics file in array for easy indexing
            metrics_labels = np.zeros(len(metrics['clusters']), dtype=np.int32)
            for i, cluster in enumerate(metrics['clusters']):
                metrics_labels[i] = cluster['label']

        elif labels is None:
            raise SyntaxError('Unit labels or metrics JSON file must be provided.')

        # Plot setup
        t = bins[:-1] + 0.5*dt_bin
        cols = 5
        rows = (len(labels) // cols) + (len(labels) % cols > 0)
        if new_fig:
            self.create_new_figure(figsize=(15, 3*rows), rows=rows, cols=cols)
        
        for i, label in enumerate(labels):
            #print('Processing label %d (%d of %d)...' % (label, i+1, len(labels)))

            # Idxs
            j = i // cols
            k = i % cols
            l = np.where(metrics_labels == label)[0][0]
            self.ax = self.axes[j, k]

            # Plot counts in stimulus window
            line = self.ax.plot(t, counts[i, :], **kwargs)[0]
            if err is not None:
                self._plot_std_area(line, err[i, :])

            # Plot average firing rate across session
            if metrics is not None:
                rate = metrics['clusters'][l]['metrics']['firing_rate']
                self.ax.plot(np.array([t[0], t[-1]]), np.array([rate]*2),
                        linestyle='--', color='C0')

            # Axis settings
            self.ax.set_title('unit %d' % label)
            self.ax.set_xlabel('time (s)')
            self.ax.set_ylabel('rate (spikes/s)')

        # Plot settings
        plt.tight_layout()
        rem = (cols - len(labels) % cols) % cols
        for i in range(1, rem+1):
            self.axes[-1, -i].axis('off')
            self.show[-1, -i] = False

    def add_trace(self, 
                  x, 
                  y, 
                  err=None, 
                  twinx=False, 
                  y_label=None, 
                  all_axes=True, 
                  **kwargs):
        if all_axes:
            axes_ = self.axes[self.show].flatten()
        else:
            axes_ = [self.ax]

        for ax_ in axes_:
            if twinx:
                ax_ = ax_.twinx()
            self.ax = ax_
            line = self.ax.plot(x, y, **kwargs)[0]
            if err is not None:
                self._plot_std_area(line, err)
            if y_label is not None:
                self.ax.set_ylabel(y_label)
        
        # Plot settings
        plt.tight_layout()

    def add_legend(self, all_axes=True, **kwargs):
        if all_axes:
            axes_ = self.axes[self.show].flatten()
        else:
            axes_ = [self.ax]

        for ax_ in axes_:
            # Get handles and labels in axis
            handles, labels = ax_.get_legend_handles_labels()
            
            # Check if axis has twin
            # https://stackoverflow.com/questions/36209575/how-to-detect-if-a-twin-axis-has-been-generated-for-a-matplotlib-axis
            twins = [a for a in self.fig.axes if a != ax_ and a.bbox.bounds == ax_.bbox.bounds]
            for twin_ax in twins:
                h, l = twin_ax.get_legend_handles_labels()
                handles += h
                labels += l
            
            # Add legend
            ax_.legend(handles, labels, **kwargs)

    def _plot_std_area(self, line, err, **kwargs):
        # Get line attributes
        x = line._xorig
        y = line._yorig
        color = line._color
        if line._alpha is None:
            alpha = 1.0
        else:
            alpha = line._alpha

        # Plot error area
        self.ax.fill_between(x, 
                             y1=y-err, 
                             y2=y+err,
                             color=color,
                             alpha=0.3*alpha,
                             **kwargs)

    def save_figure(self, filepath, ext='pdf', dpi=None):
        plt.savefig(filepath, format=ext, dpi=dpi)

