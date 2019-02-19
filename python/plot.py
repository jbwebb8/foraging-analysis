import matplotlib.pyplot as plt
import numpy as np
import math
from util import get_patch_statistics, _check_list



class Plotter:

    DEFAULT_SETTINGS = {'cmap': 'copper',
                        'style': 'seaborn-deep'}

    def __init__(self, results_dir, **kwargs):
        # Set up directory
        self.results_dir = results_dir
        
        # Plot settings
        cmap = kwargs.get('cmap', self.DEFAULT_SETTINGS['cmap'])
        self.cmap = plt.get_cmap(cmap)
        self.style = kwargs.get('style', self.DEFAULT_SETTINGS['style'])
        # TODO: initiate pyplot settings

    def _create_new_figure(self, figsize=(10, 10), rows=1, cols=1):
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
                            data,
                            days,
                            day_range=None,
                            center='median',
                            err='sem',
                            c=0.5,
                            plot_traces=False,
                            plot_points=False):

        def get_plot_idx(d):
            if day_range is None:
                return np.ones(len(d), dtype=np.bool)
            else:
                return np.logical_and(d >= day_range[0], d <= day_range[1])

        # Plot individual traces
        n_mouse = len(data)
        for i, mouse_id in enumerate(data.keys()):
            # All sessions
            if plot_traces:
                data_, days_ = get_patch_statistics(data[mouse_id], 
                                                    ids=days[mouse_id], 
                                                    method=center,
                                                    return_all=False)
                plot_idx = get_plot_idx(days_)
                self.ax.plot(days_[plot_idx], data_[plot_idx], 
                             color=self.cmap((i+1)/n_mouse))
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
                    self.ax.scatter(days_[idx], data_[idx], 
                                    color=self.cmap((i+1)/n_mouse), s=math.ceil(n/5))
            
        # Plot population trace
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
        self.ax.errorbar(days_[plot_idx], data_[plot_idx], yerr=y_err[:, plot_idx], 
                    color=self.cmap(c), capsize=5)

    def plot_harvest_rates(self, data, days, figsize=(10, 10), **kwargs):
        
        self._create_new_figure(figsize=figsize)

        self.plot_learning_curve(data, days, **kwargs)

        # Specific plot settings
        self.ax.set_xlabel('Session')
        self.ax.set_ylabel('Harvest Rate per Patch (uL/s)')
        self.ax.set_xticks(np.arange(3, 20, 2))
        

    def save_figure(self, filename, ext='pdf', dpi=None):
        plt.savefig(self.results_dir + filename, format=ext, dpi=dpi)

