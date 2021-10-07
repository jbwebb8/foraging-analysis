import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from util import get_patch_statistics, _check_list, in_interval

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

    def update_rcparams(self, rcparams):
        matplotlib.rcParams.update(rcparams)

    def set_cmap(self, name):
        self.cmap = plt.get_cmap(name)

    def create_new_figure(self, figsize=(15, 15), rows=1, cols=1, **kwargs):
        # Clear old figure
        plt.close('all')
        
        # Create figure and specified axes
        self.fig, self.axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)

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
        if isinstance(idx, list):
            idx = tuple(idx)
        self.ax = self.axes[idx]

    def set_yscale(self, scale, all_axes=True):
        if all_axes:
            axes_ = self.axes[self.show].flatten()
        else:
            axes_ = [self.ax]

        for ax_ in axes_:
            ax_.set_yscale(scale)

    def set_xscale(self, scale, all_axes=True):
        if all_axes:
            axes_ = self.axes[self.show].flatten()
        else:
            axes_ = [self.ax]

        for ax_ in axes_:
            ax_.set_xscale(scale)

    def plot_learning_curve(self,
                            days,
                            data,
                            day_range=None,
                            center='median',
                            err='sem',
                            err_plot='bar',
                            label=None,
                            c=0.5,
                            plot_traces=False,
                            plot_points=False):

        def get_plot_idx(d, mouse_id=None):
            if isinstance(day_range, dict): # dictionary of values per mouse
                if mouse_id is not None:
                    # Return values for particular mouse
                    day_range_ = day_range[mouse_id]
                else:
                    # Return values for all mice
                    plot_idx = []
                    for mouse_id_, day_range_ in day_range.items():
                        plot_idx.append(np.array(get_plot_idx(d, mouse_id_)))
                    plot_idx = (np.sum(np.vstack(plot_idx), axis=0) > 0)
                    return plot_idx
            else:
                # Otherwise, leave unchanged
                day_range_ = day_range        

            if day_range_ is None:
                return np.ones(len(d), dtype=np.bool)            
            elif len(day_range_) == 2: # assume [min, max]
                return np.logical_and(d >= day_range_[0], d <= day_range_[1])
            elif isinstance(day_range_, (list, np.ndarray)): # assume array of possible values
                return np.isin(d, np.array(day_range_))

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
                plot_idx = get_plot_idx(days_, mouse_id)
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
                plot_idx = get_plot_idx(days_, mouse_id)
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
        plot_idx = get_plot_idx(days_)
        if err_plot.lower() == 'bar':
            y_err = np.vstack([np.zeros(len(data_err)), data_err])
            self.ax.errorbar(days_[plot_idx], 
                            data_[plot_idx], 
                            yerr=y_err[:, plot_idx], 
                            color=self.cmap(c),
                            marker='o',
                            capsize=5,
                            label=label)
        elif err_plot.lower() == 'fill':
            y_err_low = data_[plot_idx] - data_err[plot_idx]
            y_err_high = data_[plot_idx] + data_err[plot_idx]
            self.ax.plot(days_[plot_idx], 
                         data_[plot_idx], 
                         color=self.cmap(c),
                         linewidth=3,
                         label=label)
            self.ax.fill_between(days_[plot_idx],
                                 y1=y_err_low,
                                 y2=y_err_high,
                                 color=self.cmap(c),
                                 alpha=0.5)

    def plot_harvest_rates(self, 
                           days, 
                           hr, 
                           figsize=(10, 10), 
                           new_fig=True, 
                           **kwargs):
        # Create new figure
        if new_fig:
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

    def plot_residence_times(self, 
                             days, 
                             t_p_obs, 
                             t_p_opt=None, 
                             figsize=(10, 10),
                             new_fig=True, 
                             **kwargs):
        # Create new figure
        if new_fig:
            self.create_new_figure(figsize=figsize)

        # Plot observed data
        self.plot_learning_curve(days, t_p_obs, label='observed', **kwargs)

        # Plot optimal data
        if t_p_opt is not None:
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
        xlim = self.ax.get_xlim()
        self.ax.set_xlim([xlim[0], xlim[1] + 5])

    def plot_travel_times(self, 
                          days, 
                          t_t, 
                          figsize=(10, 10), 
                          new_fig=True,
                          **kwargs):
        # Create new figure
        if new_fig:
            self.create_new_figure(figsize=figsize)

        # Plot observed data
        self.plot_learning_curve(days, t_t, **kwargs)

        # Plot settings
        self.ax.legend()
        self.ax.set_title('Travel Times across Sessions')
        self.ax.set_xlabel('Session')
        self.ax.set_ylabel('Travel Time (s)')
        xlim = self.ax.get_xlim()
        self.ax.set_xlim([xlim[0], xlim[1] + 5])

    def plot_session_summary(self, *,
                             T,
                             t_patch,
                             t_lick,
                             t_motor,
                             dt_chunk=250,
                             figsize=(20, 10),
                             c_patch='green',
                             c_lick='black',
                             c_motor='red'):
        # Create new figure
        self.create_new_figure(figsize=figsize)

        # Plot by time chunk
        n_chunks = math.ceil(T/dt_chunk)
        for i in range(n_chunks):
            t_start = i * dt_chunk
            t_stop = (i+1) * dt_chunk
            
            # Get patch times within chunk
            idx_start = in_interval(t_patch[:, 0], 
                                    np.asarray([t_start]), 
                                    np.asarray([t_stop]),
                                    query='event')
            idx_stop = in_interval(t_patch[:, 1], 
                                   np.asarray([t_start]), 
                                   np.asarray([t_stop]),
                                   query='event')
            idx = ((idx_start + idx_stop) > 0)
            t_patch_ = t_patch[idx, :]

            # Correct corner cases to include fraction of intervals
            if t_patch_.shape[0] > 0:
                t_patch_[0, 0] = max(t_patch_[0, 0], t_start)
                t_patch_[-1, -1] = min(t_patch_[-1, -1], t_stop-0.01) # avoids zero-ing with modular division
            
            # Plot filled rectangles to represent patches
            t_patch_ %= dt_chunk
            for j in range(t_patch_.shape[0]):
                h_patch = self.ax.fill_between(t_patch_[j], i-0.3, i+0.3, color=c_patch, alpha=0.25)

            # Plot lick times within chunk
            idx = in_interval(t_lick, 
                              np.asarray([t_start]), 
                              np.asarray([t_stop]),
                              query='event')
            t_lick_ = t_lick[idx.astype(np.bool)] % dt_chunk
            h_lick = self.ax.vlines(t_lick_, i-0.3, i+0.3, color=c_lick)
            
            # Plot motor times within chunk
            idx = in_interval(t_motor, 
                              np.asarray([t_start]), 
                              np.asarray([t_stop]),
                              query='event')
            t_motor_ = t_motor[idx.astype(np.bool)] % dt_chunk
            h_motor = self.ax.vlines(t_motor_, i-0.3, i+0.3, color=c_motor)

        # Plot settings
        self.ax.set_yticklabels(self.ax.get_yticks()*dt_chunk)
        self.ax.set_xticks([])
        self.ax.legend([h_patch, h_lick, h_motor], ['patch', 'lick', 'reward'], loc=4)
        self.ax.invert_yaxis()
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)

    def plot_psth(self, *,
                  counts,
                  bins,
                  dt_bin,
                  err=None,
                  labels=None,
                  metrics=None,
                  cols=5,
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
                self.ax.plot(np.array([t[0], t[-1]]), 
                             np.array([rate]*2),
                             linestyle='--', 
                             **{k:v for k, v in kwargs.items() if k != 'label'})

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

    def bar_graph_by_condition(self, 
                               data, 
                               cond, 
                               cond_params,
                               include_cond=None,
                               c=0.5,
                               figsize=(15, 15),
                               new_fig=True,
                               x_offset=0.0,
                               center='median',
                               err='sem',
                               plot_subjects=True,
                               plot_scatter=False):
        # Create new figure
        if new_fig:
            self.create_new_figure(figsize=figsize)
        
        # Convert to dictionary if needed (e.g. single animal data)
        if not isinstance(data, dict):
            data = {'mouse': data}
            cond = {'mouse': cond}

        # Plot combined data
        # Get consolidated data across animals
        d_plot, cond_plot = get_patch_statistics(data, 
                                            ids=cond, 
                                            method=center, 
                                            return_all=False)
        d_err, cond_plot = get_patch_statistics(data, 
                                            ids=cond, 
                                            method=center, 
                                            return_all=False)

        # Format included conditions
        if include_cond is None:
            include_cond = np.unique(cond_plot)
        elif not isinstance(include_cond, np.ndarray):
            include_cond = np.asarray(include_cond)
        include_cond = include_cond[np.isin(include_cond, np.unique(cond_plot))]
        
        # Exclude condition data
        idx = np.isin(cond_plot, include_cond)
        d_plot = d_plot[idx]
        d_err = d_err[idx]
        cond_plot = cond_plot[idx]

        # Determine plot order (patch statistics returned sorted by default)
        idx = np.argwhere(include_cond[np.newaxis, :] == cond_plot[:, np.newaxis])
        cond_order = np.argsort(idx[:, 1])
        assert (cond_plot[cond_order] == include_cond).all()
        
        # Plot consolidated data
        x = np.arange(len(cond_plot)) \
            + plot_subjects*len(data.keys())*(len(cond_plot) + 1) \
            + x_offset
        yerr = np.vstack([np.zeros(len(d_err)), d_err])
        self.ax.bar(x, 
                    d_plot[cond_order], 
                    yerr=yerr[:, cond_order],
                    color=self.cmap(c))
        
        # Format axis
        self.ax.set_xticks([]) # clear default ticks
        self.ax.set_xticklabels([]) # clear default labels
        xticks = self.ax.get_xticks()
        self.ax.set_xticks(list(xticks) + list(x - x_offset))
        xtick_labels = [label._text for label in self.ax.get_xticklabels()
                        if label._text != '']
        new_labels = [', '.join([str(p) for p in params]) for key, params in cond_params.items()
                      if key in include_cond]
        new_labels = [new_labels[i] for i in cond_order] # reorder conditions
        if plot_subjects:
            new_labels[0] = '{}\n'.format('all') + new_labels[0]
        self.ax.set_xticklabels(xtick_labels + new_labels, rotation=45, ha='right')

        # Plot data over animals, conditions
        if plot_subjects or plot_scatter:
            # Cache combined data
            d_plot_all = d_plot
            d_err_all = d_err
            cond_plot_all = cond_plot

            for i, mouse_id in enumerate(data.keys()):
                # Get all data
                d_all, cond_all = get_patch_statistics(data[mouse_id],
                                                    ids=cond[mouse_id],
                                                    return_all=True)
                
                # Get patch statistics by experimental condition
                d_plot, cond_plot = get_patch_statistics(data[mouse_id], 
                                                    ids=cond[mouse_id], 
                                                    method=center, 
                                                    return_all=False)
                d_err, cond_plot = get_patch_statistics(data[mouse_id], 
                                                    ids=cond[mouse_id], 
                                                    method=err, 
                                                    return_all=False)

                # Exclude condition data
                idx = np.isin(cond_all, include_cond)
                d_all = d_all[idx]
                cond_all = cond_all[idx]
                idx = np.isin(cond_plot, include_cond)
                d_plot = d_plot[idx]
                d_err = d_err[idx]
                cond_plot = cond_plot[idx]

                if plot_subjects:
                    # Plot statistic
                    x = np.arange(len(cond_plot)) + i*(len(cond_plot) + 1) + x_offset
                    yerr = np.vstack([np.zeros(len(d_err)), d_err])
                    self.ax.bar(x, 
                                d_plot[cond_order], 
                                yerr=yerr[:, cond_order],
                                color=self.cmap(c))

                    # Format axis
                    xticks = self.ax.get_xticks()
                    self.ax.set_xticks(list(xticks) + list(x - x_offset))
                    xtick_labels = [label._text for label in self.ax.get_xticklabels()
                                    if label._text != '']
                    new_labels = [', '.join([str(p) for p in params]) for key, params in cond_params.items()
                                  if key in include_cond]
                    new_labels = [new_labels[i] for i in cond_order] # reorder conditions
                    new_labels[0] = '{}\n'.format(mouse_id) + new_labels[0]
                    self.ax.set_xticklabels(xtick_labels + new_labels)

                if plot_scatter:
                    # Add NaN to conditions not experienced
                    if len(d_plot) < len(cond_plot_all):
                        d_plot_ = np.zeros([len(cond_plot_all)])
                        d_err_ = np.zeros([len(cond_plot_all)])
                        for j, cond_ in enumerate(cond_plot_all):
                            if cond_ not in cond_plot:
                                d_plot_[j] = np.nan
                                d_err_[j] = np.nan
                            else:
                                idx = np.argwhere(cond_plot == cond_)
                                d_plot_[j] = d_plot[idx]
                                d_err_[j] = d_err[idx]
                        d_plot = d_plot_
                        d_err = d_err_

                    # Plot statistic
                    x = np.arange(len(cond_plot_all)) \
                        + plot_subjects*len(data.keys())*(len(cond_plot_all) + 1) \
                        + x_offset
                    #x = x.astype(np.float64) + np.linspace(-0.3, 0.3, len(data.keys())+1)[i+1]
                    x = x.astype(np.float64) + np.random.uniform(0.10, 0.30)*(-1)**random.randrange(2)
                    #yerr = np.vstack([np.zeros(len(d_err)), d_err])
                    yerr =  np.vstack([d_err, d_err])
                    self.ax.errorbar(x, 
                                     d_plot[cond_order], 
                                     yerr=yerr[:, cond_order],
                                     color=self.cmap(0.0),
                                     capsize=0,
                                     marker='o',
                                     linestyle='none')

    def swarmplot_by_condition(self,
                               data, 
                               cond, 
                               cond_params,
                               include_cond=None,
                               c=0.5,
                               figsize=(15, 15),
                               new_fig=True,
                               x_offset=0.0,
                               x_spacing=None,
                               plot_subjects=True,
                               s=1.0,
                               r_factor=1.0,
                               order='ascending',
                               **kwargs):
        """s=1.0,
            r_factor=1.0,
            order='ascending',"""
        return self._scatter_by_condition(data, 
                                          cond, 
                                          cond_params,
                                          plot_type='swarm',
                                          include_cond=include_cond,
                                          c=c,
                                          figsize=figsize,
                                          new_fig=new_fig,
                                          x_offset=x_offset,
                                          x_spacing=x_spacing,
                                          plot_subjects=plot_subjects,
                                          s=s,
                                          r_factor=r_factor,
                                          order=order,
                                          **kwargs)

    def boxplot_by_condition(self,
                             data, 
                             cond, 
                             cond_params,
                             include_cond=None,
                             c=0.5,
                             figsize=(15, 15),
                             new_fig=True,
                             x_offset=0.0,
                             x_spacing=None,
                             plot_subjects=True,
                             **kwargs):
        return self._scatter_by_condition(data, 
                                          cond, 
                                          cond_params,
                                          plot_type='box',
                                          include_cond=include_cond,
                                          c=c,
                                          figsize=figsize,
                                          new_fig=new_fig,
                                          x_offset=x_offset,
                                          x_spacing=x_spacing,
                                          plot_subjects=plot_subjects,
                                          **kwargs)


    def _scatter_by_condition(self,
                              data,
                              cond,
                              cond_params,
                              plot_type,
                              include_cond=None,
                              c=0.5,
                              figsize=(15, 15),
                              new_fig=True,
                              x_offset=0.0,
                              x_spacing=None,
                              x_centers=None,
                              plot_subjects=True,
                              s=1.0,
                              r_factor=1.0,
                              order='ascending',
                              max_pts=None,
                              xlim=None,
                              ylim=None,
                              **kwargs):

        # Create new figure
        if new_fig:
            self.create_new_figure(figsize=figsize)
        
        # Convert to dictionary if needed (e.g. single animal data)
        if not isinstance(data, dict):
            data = {'mouse': data}
            cond = {'mouse': cond}
        
        # Plot combined data
        # Get consolidated data across animals
        d_plot, cond_plot = get_patch_statistics(data, 
                                                 ids=cond, 
                                                 return_all=True)

        # Format included conditions
        if include_cond is None:
            include_cond = np.unique(cond_plot)
        elif not isinstance(include_cond, np.ndarray):
            include_cond = np.asarray(include_cond)
        
        # Exclude condition data
        idx = np.isin(cond_plot, include_cond)
        d_plot = d_plot[idx]
        cond_plot = cond_plot[idx]

        # Determine plot order (patch statistics returned sorted by default)
        idx = np.argwhere(include_cond[np.newaxis, :] == np.unique(cond_plot)[:, np.newaxis])
        cond_order = np.argsort(idx[:, 1])
        i = 0
        for cond_i in np.unique(cond_plot)[cond_order]:
            while cond_i != include_cond[i]:
                i += 1
                if i == len(include_cond):
                    raise SyntaxError('Condition order not compatible with included conditions.')
        include_cond = np.unique(cond_plot)[cond_order]

        # Set swarm-specific variables
        if plot_type.lower() == 'swarm' and (xlim is None or ylim is None):
            if ylim is None:
                # Get limits from dummy plot
                self.ax.scatter(np.ones(len(d_plot)), d_plot, s=0.0)
                ylim = self.ax.get_ylim()
            if xlim is None:
                if x_centers is not None:
                    xlim = [x_centers.min() + x_offset - 0.5, 
                            x_centers.max() + x_centers.max()*plot_subjects*len(data.keys())*(len(include_cond) + 1) + x_offset + 0.5]
                else:
                    xlim = [x_offset - 0.5, len(include_cond) + plot_subjects*len(data.keys())*(len(cond_order) + 1) + x_offset + 0.5]
        
        # Get centers for each category
        if x_centers is None:
            centers = np.arange(len(include_cond)) \
                      + plot_subjects*len(data.keys())*(len(include_cond) + 1)
        else:
            assert len(x_centers) == len(include_cond)
            centers = x_centers + x_centers.max()*plot_subjects*len(data.keys())*(len(include_cond) + 1)
        centers = centers.astype(np.float64)
        if x_spacing is not None:
            if not isinstance(x_spacing, np.ndarray):
                x_spacing = np.asarray(x_spacing)
            if len(x_spacing) != len(centers):
                raise ValueError('x_spacing must be the same length as number'
                                + ' of included conditions.') 
            centers = x_spacing*centers
        centers += x_offset

        # Plot consolidated data
        for cond_i, center in zip(include_cond, centers):
            y = d_plot[cond_plot == cond_i]
            if plot_type.lower() == 'swarm':
                if max_pts is not None:
                    y = np.random.permutation(y)[:max_pts]
                x = center*np.ones(y.shape[0])
                x, y = self._apply_swarm_spacing(x, y,
                                            s=s,
                                            r_factor=r_factor,
                                            order=order,
                                            xlim=xlim,
                                            ylim=ylim)
                self.ax.scatter(x, y, 
                                color=self.cmap(c),
                                s=s,
                                **kwargs)
            elif plot_type.lower() == 'box':
                self.ax.boxplot(y,
                                positions=np.array([center]), 
                                **kwargs)
            else:
                raise ValueError('Unknown scatter type \'{}\'.'.format(plot_type))

        # Format axis
        self.ax.set_xticks([]) # clear default ticks
        self.ax.set_xticklabels([]) # clear default labels
        new_labels = [', '.join([str(p) for p in cond_params[key]]) for key in include_cond]
        if plot_subjects:
            if len(new_labels[0]) > 0:
                new_labels[0] = '{}\n'.format('all') + new_labels[0]
            else:
                new_labels[0] = 'all'
        xticks_all = list(centers - x_offset)
        xlabels_all = new_labels
        
        # Plot data over animals, conditions
        if plot_subjects:
            for i, mouse_id in enumerate(data.keys()):
                # Get all data for animal
                d_plot, cond_plot = get_patch_statistics(data[mouse_id],
                                                    ids=cond[mouse_id],
                                                    return_all=True)

                # Exclude condition data
                idx = np.isin(cond_plot, include_cond)
                d_plot = d_plot[idx]
                cond_plot = cond_plot[idx]

                # Get centers for each category
                if x_centers is None:
                    centers = np.arange(len(include_cond)) + i*(len(include_cond) + 1)
                else:
                    centers = x_centers + x_centers.max()*i*(len(include_cond) + 1)
                centers = centers.astype(np.float64)
                if x_spacing is not None:
                    centers = x_spacing*centers
                centers += x_offset

                # Plot consolidated data
                for cond_i, center in zip(include_cond, centers):
                    y = d_plot[cond_plot == cond_i]
                    if y.size > 0:
                        if plot_type.lower() == 'swarm':
                            if max_pts is not None:
                                y = np.random.permutation(y)[:max_pts]
                            x = center*np.ones(y.shape[0])
                            x, y = self._apply_swarm_spacing(x, y,
                                                        s=s,
                                                        r_factor=r_factor,
                                                        order=order,
                                                        xlim=xlim,
                                                        ylim=ylim)
                            self.ax.scatter(x, y, 
                                            color=self.cmap(c),
                                            s=s,
                                            **kwargs)

                        elif plot_type.lower() == 'box':
                            self.ax.boxplot(y, 
                                            positions=np.array([center]),
                                            **kwargs)
                        else:
                            raise ValueError('Unknown scatter type \'{}\'.'.format(plot_type))

                # Format axis
                new_labels = [', '.join([str(p) for p in cond_params[key]]) for key in include_cond]
                if len(new_labels[0]) > 0:
                    new_labels[0] = '{}\n'.format(mouse_id) + new_labels[0]
                else:
                    new_labels[0] = mouse_id
                xticks_all += list(centers - x_offset)
                xlabels_all += new_labels
        
        # Set axis limits to ensure swarmplot spacing correct
        if plot_type.lower() == 'swarm':
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        
        self.ax.set_xticks(xticks_all)
        self.ax.set_xticklabels(xlabels_all, rotation=45, ha='right')

    def _apply_swarm_spacing(self, x, y, 
                             s=1.0, 
                             r_factor=1.0, 
                             order='ascending',
                             xlim=None,
                             ylim=None):
        # Get axis limits based on dummy scatter plot if not provided
        if xlim is None or ylim is None:
            pts = self.ax.scatter(x, y, s=0.0)
            if xlim is None:
                xlim = self.ax.get_xlim()
            if ylim is None:
                ylim = self.ax.get_ylim()
        
        # Set axis limits to set transform
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Transform data coordinates to pixel coordinates
        xy_pix = self.ax.transData.transform(np.vstack([x, y]).T)
        x, y = xy_pix.T # (x, y) in pixel coordinates

        # Place pixel coordinates in specified order
        r = r_factor*0.5*(s**0.5) # convert size parameter to bounding box radius
        sort_idx = np.argsort(y)
        if order.lower() == 'descending':
            sort_idx = sort_idx[::-1]
        elif order.lower() != 'ascending':
            raise ValueError('Unknown order \'{}\'.'.format(order))
        x = x[sort_idx]
        y = y[sort_idx]

        # Next, iterate through points to determine new x-position
        for i in range(y.shape[0]):
            #print(i)
            # Determine previous points that overlap with current position
            d = ((x[i] - x[:i])**2 + (y[i] - y[:i])**2)**0.5
            #idx = np.atleast_1d(np.argwhere(d < 2*r).squeeze())
            idx = np.atleast_1d(np.argwhere(y[i] - y[:i] < 2*r).squeeze())
            #print(idx)
            
            if len(idx) > 0 and np.sum(d < 2*r) > 0:
                # Find new candidates (x', y) for x-coordinate that place current point
                # adjacent to previous point by shifting only horizontally. These
                # satisfy the equation:
                #    ((x' - a)^2 + (y - b)^2)^0.5 >= 2r
                # for all previous points (a, b). To estimate new x-coordinate, solve
                # the equation above for the new coordinate x':
                #    x' >= (4r^2 - (y - b)^2)^0.5 + a
                # accounting for both the positive and negative square root. Finally,
                # see which candidates also satisfy the minimum distance from all other
                # previously plotted points and take the one with the minimum shift
                # from the original position.
                k = 1.01 # scale factor to correct for floating point errors
                max_iters = 10 # max iterations in loop (k ~ 1.6)
                for j in range(max_iters):
                    # Find new candidates
                    x_new = np.hstack([ k*(4*(r**2) - (y[i] - y[idx]))**0.5 + x[idx],
                                       -k*(4*(r**2) - (y[i] - y[idx]))**0.5 + x[idx]])
                    dx = x_new[np.newaxis, :] - x[idx, np.newaxis]
                    dy = y[np.newaxis, i] - y[idx, np.newaxis]
                    d = (dx**2 + dy**2)**0.5
                    idx_new = np.atleast_1d(np.argwhere(np.sum(d < 2*r, axis=0) == 0).squeeze())
                    
                    # Break if 1+ candidates found
                    if idx_new.size > 0:
                        break
                    else:
                        k = 1.05*k

                # Select new x-coordinate as minimal shift from current position
                x[i] = x_new[idx_new][np.argmin(np.abs(x[i] - x_new[idx_new]))]

        # Transform pixel coordinates back to data coordinates
        xy_data = self.ax.transData.inverted().transform(np.vstack([x, y]).T)
        x_data, y_data = xy_data.T

        return x_data, y_data

        

    def plot_by_condition(self, 
                          data, 
                          cond, 
                          cond_params,
                          include_cond=None,
                          center='median',
                          err='sem',
                          c=0.5,
                          figsize=(15, 15),
                          new_fig=True,
                          plot_subjects=True,
                          capsize=None,
                          markersize=None,
                          linewidth=1.0):
        # Create new figure
        if new_fig:
            self.create_new_figure(figsize=figsize)
        
        # Convert to dictionary if needed (e.g. single animal data)
        if not isinstance(data, dict):
            data = {'mouse': data}
            cond = {'mouse': cond}

        # Format included conditions
        if include_cond is None:
            include_cond = np.unique(cond_plot)
        elif not isinstance(include_cond, np.ndarray):
            include_cond = np.asarray(include_cond)

        # Determine plot order (patch statistics returned sorted by default)
        idx = np.argwhere(include_cond[np.newaxis, :] == cond_plot[:, np.newaxis])
        cond_order = np.argsort(idx[:, 1])
        assert (cond_plot[cond_order] == include_cond).all()
        
        # Plot data over animals, conditions
        if plot_subjects:
            for i, mouse_id in enumerate(data.keys()):
                # Get all data
                d_all, cond_all = get_patch_statistics(data[mouse_id],
                                                    ids=cond[mouse_id],
                                                    return_all=True)

                # Get patch statistics by experimental condition
                d_plot, cond_plot = get_patch_statistics(data[mouse_id], 
                                                    ids=cond[mouse_id], 
                                                    method=center, 
                                                    return_all=False)
                d_err, cond_plot = get_patch_statistics(data[mouse_id], 
                                                    ids=cond[mouse_id], 
                                                    method=err, 
                                                    return_all=False)

                # Exclude condition data
                idx = np.isin(cond_all, include_cond)
                d_all = d_all[idx]
                cond_all = cond_all[idx]
                idx = np.isin(cond_plot, include_cond)
                d_plot = d_plot[idx]
                d_err = d_err[idx]
                cond_plot = cond_plot[idx]
                
                # Plot statistic
                x = np.arange(len(cond_plot))
                self.ax.plot(x, 
                        d_plot[cond_order], 
                        color=self.cmap((i)/len(data.keys())),
                        linewidth=linewidth,
                        marker='o',
                        markersize=markersize,
                        label=mouse_id)

        # Plot combined data  
        # Get consolidated data across animals
        d_plot, cond_plot = get_patch_statistics(data, 
                                            ids=cond, 
                                            method=center, 
                                            return_all=False)
        d_err, cond_plot = get_patch_statistics(data, 
                                            ids=cond, 
                                            method=err, 
                                            return_all=False)

        # Exclude condition data
        idx = np.isin(cond_plot, include_cond)
        d_plot = d_plot[idx]
        d_err = d_err[idx]
        cond_plot = cond_plot[idx]

        # Plot statistic
        x = np.arange(len(cond_plot))
        yerr = d_err
        self.ax.errorbar(x, 
                    d_plot[cond_order], 
                    yerr=yerr[cond_order],
                    color=self.cmap(0.0),
                    linewidth=2*linewidth,
                    capsize=capsize,
                    marker='o',
                    markersize=markersize,
                    label='all')

        if plot_subjects:
            # Add legend, removing error bar
            handles, labels = self.ax.get_legend_handles_labels()
            handles = [h for h in handles[:-1]] + [handles[-1][0]]
            self.ax.legend(handles, labels, markerscale=0.1)

        # Set axis labels
        self.ax.set_xticks(x)
        new_labels = [', '.join([str(p) for p in params]) for key, params in cond_params.items()
                      if key in include_cond]
        new_labels = [new_labels[i] for i in cond_order] # reorder conditions
        self.ax.set_xticklabels(new_labels, rotation=45, ha='right')

    def _format_cond_order(self, cond_order, cond_params, exclude_cond):
        if cond_order is None:
            cond_order = np.arange(len(cond_params) - len(exclude_cond))
        else:
            assert len(cond_order) == len(cond_params) - len(exclude_cond)
            if not isinstance(cond_order, np.ndarray):
                cond_order = np.array(cond_order)
            gt = (cond_order[np.newaxis, :] > np.array(exclude_cond)[:, np.newaxis])
            cond_order -= np.sum(gt, axis=0)
        
        return cond_order

    def save_figure(self, filepath, ext='pdf', dpi=None):
        plt.savefig(filepath, format=ext, dpi=dpi)

