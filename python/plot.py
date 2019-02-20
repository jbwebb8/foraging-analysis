import matplotlib.pyplot as plt
import numpy as np
import math
from util import get_patch_statistics, _check_list

class Plotter:

    DEFAULT_SETTINGS = {'cmap': 'copper',
                        'style': 'seaborn-deep'}

    def __init__(self, **kwargs):
        # Plot settings
        cmap = kwargs.get('cmap', self.DEFAULT_SETTINGS['cmap'])
        self.cmap = plt.get_cmap(cmap)
        self.style = kwargs.get('style', self.DEFAULT_SETTINGS['style'])
        # TODO: initiate pyplot settings

    def create_new_figure(self, figsize=(10, 10), rows=1, cols=1):
        # Create figure and specified axes
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize)
        
        # General plot settings
        for ax in _check_list(self.axes):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Set current axis to first if multiple axes
        if (rows == 1) and (cols == 1):
            self.ax = self.axes
        elif (rows == 1) or (cols == 1):
            self.ax = self.axes[0]
        else:
            self.ax = self.axes[0, 0]

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
                             label=mouse_id)
            if plot_points:
                data_, days_ = get_patch_statistics(data[mouse_id], 
                                                    ids=days[mouse_id], 
                                                    method=center,
                                                    return_all=True)
                plot_idx = get_plot_idx(days_)
                (days_, data_) = (days_[plot_idx], data_[plot_idx])
                for dt in np.unique(data_):
                    idx = (data_ == dt)
                    n = np.sum(idx)
                    self.ax.scatter(days_[idx], 
                                    data_[idx], 
                                    color=self.cmap((i+1)/n_mouse),
                                    label=mouse_id)
            
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
        self.ax.set_xticks(np.arange(3, 20, 2))
        
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
        self.plot_learning_curve(hr_diff_opt, days, label='observed vs. optimal', **kwargs)

        # Plot difference between observed and max
        hr_diff_max = {}
        for [mouse_id, hr_max_mouse], [mouse_id, hr_obs_mouse] in zip(hr_max.items(), hr_obs.items()):
            hr_diff_max[mouse_id] = []
            for hr_max_i, hr_obs_i in zip(hr_max_mouse, hr_obs_mouse):
                hr_diff_max[mouse_id] = hr_obs_i - hr_max_i
        self.plot_learning_curve(hr_diff_max, days, label='observed vs. maximum', **kwargs)

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
        if day_range is None:
            plot_idx = np.ones(len(days), dtype=np.bool)
        else:
            plot_idx = np.logical_and(days >= day_range[0], days <= day_range[1])
        t_p_opt_, days_ = get_patch_statistics(t_p_opt, 
                                               ids=days, 
                                               method=center,
                                               return_all=False)
        ax.plot(days_[plot_idx], t_p_opt_[plot_idx], 
                linestyle='--', color=cmap(0.10), label='optimal')

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



    def save_figure(self, filepath, ext='pdf', dpi=None):
        plt.savefig(filepath, format=ext, dpi=dpi)

