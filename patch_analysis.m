%% Set parameters and create filelist
% Arguments:
% - filelist: txt file containing file names of all training days of one 
%   mouse, in numerical order
% - plot_data: plot file data (e.g. power vs. time) while processing
% - use_sound: use sound waveform for defining patches if position not available
filelist = 'matlist.txt';
plot_data = false;
%use_sound = false;
stop_thresh = 0.1;
run_thresh = 0.5;
fig_id = 1;

% Sort filelist and remove irrelevant filenames
[filelist, training_days] = sort_training_files(filelist);
start_idx = regexp(filelist{1}, 'j[0-9][a-z][0-9]_d');
mouse_id = filelist{1}(start_idx:start_idx+3);

%% Extract patch data
% Settings
fft_pts = 256;
lp_pass = 14; % Fp in Hz
lp_stop = 15; % Fst in Hz
thresh = 0.03; % variance threshold for pink noise (TODO: update to new auto code)
save_data = false;
save_fig = false;

% Set data placeholders for each experiment
t_p = cell(size(filelist, 1), 1); % residence time
t_t = cell(size(filelist, 1), 1); % travel time
r_p = cell(size(filelist, 1), 1); % total reward per patch
d_next_patch = cell(size(filelist, 1), 1); % distance to next patch from stopping points
d_config = zeros(size(filelist, 1), 2); % [d_patch, d_interpatch]
keep_idx = ones(size(filelist, 1), 1); % files to include in analysis

tic
for i = 1:size(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    pe = PatchExperiment(filename);
    
    % Get patch statistics
    try
        [t_switch, in_patch] = pe.get_patch_times_from_sound(...
                fft_pts, lp_pass, lp_stop, thresh);
        [t_p{i}, t_t{i}] = pe.get_patch_times(t_switch, in_patch);
        r_p{i} = pe.get_patch_rewards(); % does NOT work with zero reward decay
        d_next_patch{i} = pe.stopping_distances(stop_thresh, run_thresh);
        d_config(i, :) = [pe.d_patch, pe.d_interpatch];
    catch ME
        if strcmp(ME.identifier, 'PatchExperiment:VarNotFound')
            msg = '%s not found in file %s. Skipping entry.';
            var_name = ME.message(10:regexp(ME.message, ' not found')-1); % hack to get variable name
            disp(fprintf(msg, var_name, filelist{i}));
            keep_idx(i) = 0;
        else 
            rethrow(ME)
        end
    end
    
    toc
end

% Remove excluded files
keep_idx = find(keep_idx);
filelist = filelist(keep_idx);
training_days = training_days(keep_idx);
t_p = t_p(keep_idx);
t_t = t_t(keep_idx);
r_p = r_p(keep_idx);
d_next_patch = d_next_patch(keep_idx);
d_config = d_config(keep_idx, :);

% Save data
if save_data
    [start_idx, end_idx] = regexp(filename, 'j[0-9]+[a-z][0-9]+_');
    %new_filename = [filename(start_idx:end_idx), 'patch_data.mat'];
    base_name = filename(1:end_idx);
    new_filename = [base_name, 'patch_data.mat'];
    fprintf('Saving to %s\n', new_filename);
    save(new_filename, 't_p', 't_t', 'r_p', 'd_next_patch', 'd_config', 'filelist');
end

% Plot patch residence times
fig1 = figure(fig_id);
clf(fig1);
hold on;
for i = 1:length(t_p)
    c1 = [0.122 0.467 0.706]; % pyplot C0 = blue
    c2 = [0.0 0.0 0.0]; % black data points
    bar(i, mean(t_p{i})/1000, 'FaceColor', c1); % mean
    errorbar(i, mean(t_p{i})/1000, 0.0, std(t_p{i})/1000, 'Color', c1); % pos std error
    scatter(i*ones(length(t_p{i}), 1), t_p{i}/1000, 20, c2, 'filled'); % data points
end
hold off;
title('Patch Residence Time');
xlabel('Training Day');
xticks(1:length(filelist));
xticklabels(training_days);
ylabel('Time (s)');
xlim([0 length(t_p)+1]);
fig_id = fig_id + 1;

% Plot travel times
fig2 = figure(fig_id);
clf(fig2);
hold on;
for i = 1:length(t_t)
    c1 = [0.122 0.467 0.706]; % pyplot C0 = blue
    c2 = [0.0 0.0 0.0]; % black data points
    bar(i, mean(t_t{i})/1000, 'FaceColor', c1); % mean
    errorbar(i, mean(t_t{i})/1000, 0.0, std(t_t{i})/1000, 'Color', c1); % pos std error
    scatter(i*ones(length(t_t{i}), 1), t_t{i}/1000, 20, c2, 'filled'); % data points
end
hold off;
title('Travel Time Between Patches');
xlabel('Training Day');
xticks(1:length(filelist));
xticklabels(training_days);
ylabel('Time (s)');
xlim([0 length(t_t)+1]);
fig_id = fig_id + 1;

% Plot average patch rewards
fig3 = figure(fig_id);
clf(fig3);
hold on;
for i = 1:length(r_p)
    c1 = [0.122 0.467 0.706]; % pyplot C0 = blue
    c2 = [0.0 0.0 0.0]; % black data points
    bar(i, mean(r_p{i}), 'FaceColor', c1); % mean
    errorbar(i, mean(r_p{i}), 0.0, std(r_p{i}), 'Color', c1); % pos std error
    scatter(i*ones(length(r_p{i}), 1), r_p{i}, 20, c2, 'filled'); % data points
end
hold off;
title('Average Cumulative Reward per Patch');
xlabel('Training Day');
xticks(1:length(filelist));
xticklabels(training_days);
ylabel('Reward (uL)');
xlim([0 length(r_p)+1]);
fig_id = fig_id + 1;

% Plot distribution of stopping points
fig4 = figure(fig_id);
clf(fig4);
hold on;
for i = 1:length(d_next_patch)
    idx_patch = d_next_patch{i}>=0 & d_next_patch{i}<=d_config(i, 1);
    idx_interpatch = d_next_patch{i}<0 | d_next_patch{i}>d_config(i, 1);
    c = zeros(length(d_next_patch{i}), 3);
    c(idx_interpatch, 1) = 0.60;
    c(idx_patch, 2) = 0.33;
    scatter(d_next_patch{i}, ones(length(d_next_patch{i}), 1) * i, [], c);
end
hold off;
title('Stopping Points during Training');
xlabel('Distance (cm)');
ylabel('Training Day');
yticks(1:length(filelist));
yticklabels(training_days);
xlim([-max(d_config(:, 2)/2)*1.1 max(d_config(:, 1)+d_config(:, 2)/2)*1.1]);
ylim([0 length(d_next_patch)+1]);
fig_id = fig_id + 1;

% Plot harvest rates
% TODO: combine this with cumulative reward plot
fig5 = figure(fig_id);
clf(fig5);
hold on;
for i = 1:length(r_p)
    c1 = [0.122 0.467 0.706]; % pyplot C0 = blue
    c2 = [0.0 0.0 0.0]; % black data points
    h_p = r_p{i} / ((t_p{i} + t_t{i})/1000);
    bar(i, mean(h_p), 'FaceColor', c1); % mean
    errorbar(i, mean(h_p), 0.0, std(h_p), 'Color', c1); % pos std error
    scatter(i*ones(length(h_p), 1), h_p, 20, c2, 'filled'); % data points
end
hold off;
title('Average Harvest Rate per Patch');
xlabel('Training Day');
xticks(1:length(filelist));
xticklabels(training_days);
ylabel('Reward (uL) / s');
xlim([0 length(r_p)+1]);
fig_id = fig_id + 1;

% Save figures if specified
if save_fig
    saveas(fig1, [base_name, 't_p']);
    saveas(fig2, [base_name, 't_t']);
    saveas(fig3, [base_name, 'r_p']);
    saveas(fig4, [base_name, 'd_stop']);
end

%% Extract basic learning metrics (d`, bias, lick rate)
% Set up matrices
var_names = {'UntitledS_HitRate', 'UntitledS_FARate', 'UntitledS_dPrime', 'UntitledS_c'};
vars = cell(length(var_names), 1);
vars(:) = {zeros(length(filelist), 2)}; % must match order above!
keep_idx = zeros(length(filelist), length(var_names));

for i = 1:length(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    pe = PatchExperiment(filename);
    
    % Get metrics. Must use try-catch statements since a few files do not
    % have variable (for some unknown reason).
    for j = 1:length(var_names)
        try
            struct = pe.load_var(var_names{j}, false);
            vars{j}(i, :) = [training_days(i), struct.Data];
            keep_idx(i, j) = 1;
        catch ME
            if strcmp(ME.identifier, 'PatchExperiment:VarNotFound')
                msg = '%s not found in file %s. Skipping entry.';
                disp(fprintf(msg, var_names{j}, filelist{i}));
            else 
                rethrow(ME)
            end
        end
    end
end

% Filter out missing days
for j = 1:length(var_names)
    vars{j} = vars{j}(find(keep_idx(:, j)), :);
end

% Plot results
for j = 1:length(var_names)
    fig = figure(fig_id);
    clf(fig);
    plot(vars{j}(:, 1), vars{j}(:, 2));
    %ylim([-2 2]);
    idx = regexp(var_names{j}, '_');
    var_name = [var_names{j}(1:idx-1), '\', var_names{j}(idx:end)];
    title(mouse_id);
    ylabel(var_name);
    xlabel('Training Day');
    fig_id = fig_id + 1;
end

%% Analyze relationship between behavior and time
% Does performance worsen with time and/or reward accrued? In order to
% answer, let's plot a sliding window of the average hit rate, etc. over
% the last n trials vs. total reward accrued.

% Params
m = 30; % length of sliding window (# trials)
n = 4; % minimum # of trials to calculate metric

% Set up matrices
hit_rate = cell(length(filelist), 1);
fa_rate = cell(length(filelist), 1);
d_prime = cell(length(filelist), 1);
c = cell(length(filelist), 1);
r_cum = cell(length(filelist), 1);

% Iterate through sessions
for i = 1:length(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    pe = PatchExperiment(filename);
    
    % Get cumulative rewards over trials
    struct = pe.load_var('UntitledRewarduL', false);
    rewards = struct.Data;
    r_cum{i} = cumsum(rewards);
    
    % Get trial data and set up matrices, such that
    % a(1:num_trials) = sliding value, a(end) = global value
    struct = pe.load_var('UntitledTrialResult', false);
    results = struct.Data;
    num_trials = length(results);
    hit_rate{i} = zeros(num_trials + 1, 1);
    fa_rate{i} = zeros(num_trials + 1, 1);
    d_prime{i} = zeros(num_trials + 1, 1);
    c{i} = zeros(num_trials + 1, 1);
    
    % Calculate sliding values: slide along trial outcomes with window of size m
    for j = 1:num_trials
        % Get (at most) m recent results
        results_m = results(max(1, j-m+1):j);
        
        % Calculate metrics for window
        [hit_rate{i}(j), fa_rate{i}(j), d_prime{i}(j), c{i}(j)] = ...
            calculate_sdt_metrics(results_m, n);
    end
    
    % Calculate global values
    [hit_rate{i}(end), fa_rate{i}(end), d_prime{i}(end), c{i}(end)] = ...
            calculate_sdt_metrics(results, n);
end

% Plot results
% Clear figures
for i = 1:4
    clf(figure(fig_id + i - 1));
    hold on;
end

% Plot sliding values as line, global value as dot on right
legend_obs = gobjects(length(filelist), 4);
for i = 1:length(filelist)
    figure(fig_id);
    p = plot(hit_rate{i}(1:end-1));
    legend_obs(i, 1) = p(1);
    ax = gca;
    ax.ColorOrderIndex = i;
    plot(1:length(hit_rate{i}), ones(length(hit_rate{i}), 1)*hit_rate{i}(end), ...
        'LineStyle', '--');
    
    figure(fig_id + 1);
    p = plot(fa_rate{i}(1:end-1));
    legend_obs(i, 2) = p(1);
    ax = gca;
    ax.ColorOrderIndex = i;
    plot(1:length(fa_rate{i}), ones(length(fa_rate{i}), 1)*fa_rate{i}(end), ...
        'LineStyle', '--');
    
    figure(fig_id + 2);
    p = plot(d_prime{i}(1:end-1));
    legend_obs(i, 3) = p(1);
    ax = gca;
    ax.ColorOrderIndex = i;
    plot(1:length(d_prime{i}), ones(length(d_prime{i}), 1)*d_prime{i}(end), ...
        'LineStyle', '--');
    
    figure(fig_id + 3);
    p = plot(c{i}(1:end-1));
    legend_obs(i, 4) = p(1);
    ax = gca;
    ax.ColorOrderIndex = i;
    plot(1:length(c{i}), ones(length(c{i}), 1)*c{i}(end), ...
        'LineStyle', '--');
end

% Label plots
y_labels = {'Hit Rate', 'FA Rate', 'd`', 'c'};
for i = 1:4
    figure(fig_id+i-1);
    title(mouse_id);
    xlabel('Trial #');
    ylabel(y_labels{i});
    legend(legend_obs(:, i), "Day " + string(training_days));
    %legend("Day " + string(training_days));
end

hold off;
fig_id = fig_id + 4;

%% Analyze velocity traces
% Let's look at the average velocity trace wrapped over distance relative
% to patches.

% Parameters
filter_len = 1000; % length of sliding filter (ms)
bin_size = 5; % size of bin for calculating velocity average (cm)

for i = 1:length(filelist)
    % Load session
    filename = filelist{i};
    pe = PatchExperiment(filename);

    % Grab variables
    v = pe.wheel_speed(filter_len);
    x = pe.linear_position(filter_len);
    %plot(v); hold on; plot(x*max(v)/max(x)); hold off; % check for alignment
    d_block = pe.d_patch + pe.d_interpatch;
    x = mod(x, d_block); % wrap position
    num_bins = ceil(d_block / bin_size); % number of bins of size bin_size

    % Get velocity mean and median within each bin
    v_bin = cell(num_bins, 1);
    v_mean = zeros(num_bins, 1);
    v_med = zeros(num_bins, 1);
    for j = 1:num_bins
        low = (j-1) * bin_size;
        high = j * bin_size;
        idx = ((x >= low) + (x < high)) == 2; % AND operation
        v_bin_j = v(idx);
        v_bin{j} = v_bin_j;
        v_mean(j) = sum(v_bin_j) / length(v_bin_j);
        v_med(j) = v_bin_j(round(length(v_bin_j) / 2));
    end
    
    % Plot results 
    patch_in_middle = true; % center coordinates such that patch begins at zero
    xx = (bin_size:bin_size:d_block) - (bin_size / 2); % coordinates for bins
    if patch_in_middle
        x = x - pe.d_interpatch/2; % shift back first 1/2 of interpatch
        x(x<0) = x(x<0) + d_block; % place first 1/2 of interpatch at end
        x = x - pe.d_interpatch/2; % zero out beginning of patch
        xx = xx - pe.d_interpatch/2;
        xx(xx<0) = xx(xx<0) + d_block;
        xx = xx - pe.d_interpatch/2;
        [xx, sort_idx] = sort(xx);
        v_mean = v_mean(sort_idx);
        v_med = v_med(sort_idx);
    end
    figure(fig_id);
    clf;
    hold on;
    p1 = scatter(x, v, 2, 'Marker', '.'); % removes artifact
    p1.MarkerFaceAlpha = 0.02; % more transparent
    p1.MarkerEdgeAlpha = 0.02;
    p2 = plot(xx, v_mean);
    p3 = plot(xx, v_med);
    p4 = plot([0, 0], [min(v), max(v)], ...
        'Color', [0.466, 0.674, 0.188], 'LineStyle', '--');
    p5 = plot([pe.d_patch, pe.d_patch], [min(v), max(v)], ...
        'Color', [0.466, 0.674, 0.188], 'LineStyle', '--');
    title(sprintf('%s - Day %d', mouse_id, training_days(i)));
    xlabel('Distance (cm)');
    ylabel('Velocity (cm/s)');
    LH = [plot(nan, nan, 'Color', p1.CData*p1.MarkerFaceAlpha), ...
          plot(nan, nan, 'Color', p2.Color), ...
          plot(nan, nan, 'Color', p3.Color), ...
          plot(nan, nan, 'Color', p4.Color, 'LineStyle', '--')];
    L = {'all', 'mean', 'median', 'patch'};
    legend(LH, L);
    hold off;
    fig_id = fig_id + 1;
end

%% Analyze SDT statistics via sound waveform
% After this is validated, let's make it a separate function that the above 
% section that uses to calculate these statistics, with the option of using
% either the sound waveform (here) or the trial results (as above). 

% Settings
verbose = true; % plot each step for understanding
filename = 'j3z5_d16_2018_07_25_14_23_22.mat';
fft_pts = 256; % size of window for fft
lp_stop = 15; %Fst in Hz
lp_pass = 14; %Fp in Hz
var_thresh = 0.03; % variance threshold for pink noise
window = [3.0, 5.0]; % filter to [w1, end-w2] in seconds

% Load sound data
pe = PatchExperiment(filename);
sound_wf = pe.load_var('UntitledSound[1-4]', true);
dt_wf = sound_wf.Property.wf_increment;
window_wf = max(round(window / dt_wf), 1);
sound_wf = sound_wf.Data(window_wf(1):end-window_wf(2)); % remove end effects

% Get spectrogram of sound waveform
s = spectrogram(sound_wf, fft_pts); % complex frequency content; shape=[freq, time]
s_abs = abs(s); % complex magnitude; shape=[freq, time]
dt_s = dt_wf * fft_pts / 2;

% Create filter
d = fdesign.lowpass('Fp,Fst,Ap,Ast', ... % format
                    2*lp_pass*dt_wf*fft_pts, ... % Fp
                    2*lp_stop*dt_wf*fft_pts, ... % Fst
                    1, 60); % Ap, Ast
%designmethods(d) % lists design choices
Hd = design(d, 'butter'); % Butterworth IIR filter
if verbose
    fvtool(Hd); % visualize filter
end

% Calculate variance in frequency domain at each time point
if verbose
    % Spectrogram of complex magnitudes
    figure(fig_id);
    subplot(3, 2, 1);
    imagesc(s_abs(:, 1:10000));
    colorbar;

    % Spectrogram smoothed with median filter
    subplot(3, 2, 2);
    s_abs_filt = medfilt1(s_abs, 1); % Does first order do anything?
    imagesc(s_abs_filt(:, 1:10000));
    colorbar;

    % Differences between adjacent frequency magnitudes at each time point
    % (i.e. rough derivative in frequency domain)
    subplot(3, 2, 3);
    s_diff = diff(s_abs_filt, 1);
    imagesc(s_diff(:, 1:10000));
    colorbar;

    % Absolute values of differences
    subplot(3, 2, 4);
    s_diff_abs = abs(s_diff);
    imagesc(s_diff_abs(:, 1:10000));
    colorbar;

    % Median difference at each time point
    subplot(3, 2, 5);
    s_var = median(s_diff_abs);
    plot(s_var(1:10000));
    
    % Increment for next figure
    fig_id = fig_id + 1;
else
    % One-line calculation
    s_var = median(abs(diff(medfilt1(s_abs,1),1))); %#ok<UNRCH>
end

% Smooth median variance
s_var_filt = filtfilt(Hd.sosMatrix, Hd.ScaleValues, s_var); % zero-phase filtering
if verbose
    % Settings
    nbins = 10000;
    
    % Histogram and plot of median variance
    figure(fig_id);
    subplot(2,2,1);
    hist(s_var, nbins);
    subplot(2,2,2);
    plot(s_var(1:10000));

    % Histogram and plot of smoothed median variance
    subplot(2,2,3)
    plot(s_var, 'Marker', '.', 'LineStyle', 'none');
    hold all;
    plot(s_var_filt, 'Marker', '.', 'LineStyle', 'none');
    subplot(2,2,4);
    hist(s_var_filt, nbins);
    
    % Increment for next figure
    fig_id = fig_id + 1;
end

% Find patch entry and exit points
in_patch = s_var_filt < var_thresh; % true if in patch at time t
t_switch = find((in_patch + circshift(in_patch, -1)) == 1);

% Plot sound and in patch definitions
figure(fig_id);
clf(fig_id);
scatter(find(in_patch)*dt_s, s_var_filt(in_patch), 2);
hold on;
scatter(find(~in_patch)*dt_s, s_var_filt(~in_patch), 2);
scatter(t_switch*dt_s, ones(length(t_switch), 1)*var_thresh);
hold off;
legend('in patch', 'out of patch', 'patch entry/exit');
xlabel('Time (s)');
ylabel('Variance');
fig_id = fig_id + 1;

%% Calculate patch residence and travel times
start_in_patch = in_patch(1);
end_in_patch = xor((rem(length(t_switch), 2) == 1), start_in_patch);
if start_in_patch
    t_switch = [0 t_switch]; % insert zero at beginning for patch entry   
end
if end_in_patch
    last_entry = t_switch(end); % hold value for travel time
    t_switch = t_switch(1:end-1); % drop last patch entry
end

% patch residence time = t_exit - t_enter
t_p = reshape(t_switch, 2, [])';
t_p = t_p(:, 2) - t_p(:, 1);

% Get travel times between patches by inference. 
t_t = circshift(t_switch, -1);
if end_in_patch
    t_t(end) = last_entry; % recover last entry point
else
    t_t = t_t(1:end-2); % drop last inter-patch
end
t_t = reshape(t_t, 2, [])';
t_t = t_t(:, 2) - t_t(:, 1);

% Convert to seconds
t_p = t_p * dt_s;
t_t = t_t * dt_s;

% Filter infeasible times
% TODO: move this to a while loop to calculate for t_switch, in_patch
%v_max = max(pe.wheel_speed(500)); % cm/s
%t_p_min = (pe.d_patch / v_max); % s
%t_t_min = (pe.d_interpatch / v_max); % s
%t_p_filt = t_p(t_p > t_p_min);
%t_t_filt = t_t(t_t > t_t_min);

if verbose
    figure(fig_id);
    clf(fig_id);
    subplot(2, 1, 1);
    hold on;
    plot(t_p);
    plot(t_p_filt, 'LineStyle', '--');
    title('Patch Residence Times');
    xlabel('Patch #');
    ylabel('Time (s)');
    legend(['t\_p', 't\_p\_filt']);

    subplot(2, 1, 2);
    plot(t_t);
    hold on;
    plot(t_p_filt, 'LineStyle', '--');
    title('Travel Times');
    xlabel('Interpatch #');
    ylabel('Time (s)');
    legend(['t\_t', 't\_t\_filt']);
    
    fig_id = fig_id + 1;
end

%% Associate each trial result with timestamp and patch number
% We will utilize the sound waveform, lick trace, and motor trace (as a 
% surrogate for reward) in order to determine when each of the trial types 
% occurred:
% - Hit: +target, +lick, +motor
% - Miss: +target, -lick, -motor
% - FA: -target, +lick, -motor
% - Catch: -target, -lick, +motor
% Also, all these must be occurring during the interval with tone cloud,
% i.e. in the patch.

% Settings
fft_pts = 512; % size of window and number of frequency bins
overlap = 256; % size of overlap between adjacent windows
range = [1, 3e6]; % range of sound waveform to analyze
f_sample = 1 / dt_wf; % sampling rate of sound waveform

% Get target frequency times
struct = pe.load_var('UntitledS_TarFreqHz', false);
f_target = struct.Data;

% wavelet spectrogram, multi-taper spectrogram
% Plot PSD (estimated from FFT) of sound waveform
figure(fig_id);
p1 = subplot(3, 1, 1);
%spectrogram(sound_wf(range(1):range(2)), fft_pts, overlap, fft_pts, f_sample, 'yaxis');
[s2, f2, t2] = spectrogram(sound_wf(range(1):range(2)), ...
                        fft_pts, ...
                        overlap, ...
                        fft_pts, ...
                        f_sample);
surf(t2, f2/1000, log10(abs(s2).^2), 'EdgeColor', 'none'); % plot estimate of PSD
axis xy; axis tight; view(0,90); % make similar to spectrogram() fn
xlabel('Time (s)');
ylabel('Frequency (kHz)');
colorbar;
hold on;
plot(xlim, [f_target/1000, f_target/1000], 'Color', 'black', 'LineStyle', '--');
hold off;
xl = xlim;

% Plot PSD in target frequency band
p2 = subplot(3, 1, 2);
%fbin_target = round(f_target / ((f_sample/2) / floor(fft_pts/2 + 1))) + 1;
[~, fbin_target] = min(abs(f_target - f2)); % bin closest to target frequency
s2_abs = abs(s2); % complex magnitude
num_tbins = floor( ((range(2)-range(1))-overlap)/(fft_pts-overlap) );
t_s2 = linspace(0, range(2)-range(1), num_tbins) * dt_wf;
plot(t_s2, s2_abs(fbin_target, :));
xlim(xl);
linkaxes([p1, p2], 'x');
p3 = subplot(3, 1, 3);
histogram(s2_abs(fbin_target, :), 100);
set(gca, 'YScale', 'log');

% Align plots; get() returns [x, y, width, height]
pos1 = get(p1, 'Position'); 
pos2 = get(p2, 'Position');
pos3 = get(p3, 'Position');
pos2(3) = pos1(3);
pos3(3) = pos1(3);
set(p2, 'Position', pos2);
set(p3, 'Position', pos3);

fig_id = fig_id + 1;

%% Set threshold probability of playing target frequency

% Settings
p_thresh = 1e-10; % threshold of selecting target frequency for tc vs. target
power_thresh = 2.5; % threshold of target frequency being played
t_thresh = 0.5; % threshold of target tone duration (s)

% Find numbers of bins corresponding to chance probability
df_s2 = f2(2) - f2(1); % frequency bin size in spectrogram
struct = pe.load_var('Settings', false);
f_low = str2double(struct.Property.SoundConfigurationToneCloudConfigLowFreqHz);
f_high = str2double(struct.Property.SoundConfigurationToneCloudConfigHiFreqHz);
n = round((f_high - f_low)/df_s2); % # of frequency bins in tone cloud
k = str2double(struct.Property.SoundConfigurationToneCloudConfigFreqNumOctave); % # of frequencies per chord

p_f = k / n; % probability of choosing target freq bin for chord
dt_s2 = t2(2) - t2(1); % time bin size in spectrogram
t_chord = str2double(struct.Property.SoundConfigurationToneCloudConfigTimeBinWidthms); % duration of chord (ms)
m = (t_chord/1000)/dt_s2; % number of bins per chord
bin_thresh = ceil(m * log(p_thresh) / log(p_f)); % # of bins corresponding to p_thresh

% Find time points during which target frequency power > threshold
idx_target = s2_abs(fbin_target, :) > power_thresh;
idx_start = find((idx_target - circshift(idx_target, 1)) == 1);
idx_end = find((idx_target - circshift(idx_target, 1)) == -1);
idx_target = [idx_start; idx_end]'; % [start, end]
t_duration = (idx_end - idx_start) * dt_s2;
idx_target = idx_target(t_duration > t_thresh, :); % filter by duration
t_target = idx_target * dt_s2; % convert to time (s)

% Plot target start times
figure(fig_id);
plot(t_s2, s2_abs(fbin_target, :));
hold on;
scatter(t_s2(idx_target), power_thresh*ones(numel(idx_target), 1));
hold off;
fig_id = fig_id + 1;

%% Find lick times
lick = pe.load_var('UntitledLick[1-4]', true);
dt_lick = lick.Property.wf_increment;
window_lick = max(round(window / dt_lick), 1);
lick = lick.Data(window_lick(1):end-window_lick(2));
lick_thresh = 1.0;
t_lick = find( (lick > lick_thresh) - circshift((lick > lick_thresh), 1) == 1);
t_lick = t_lick(2:end) * dt_lick;

%% Is lick time accurate?
lick_sensor = pe.load_var('UntitledLickSensor', false);
lick_sensor = lick_sensor.Data;
lick_time = pe.load_var('UntitledLickTime', false);
lick_time = lick_time.Data;

figure(fig_id);
scatter(lick_time(1:1e6), lick_sensor(1:1e6));
hold on;
plot(0:dt_lick:1400, lick(1:1400/dt_lick+1));
hold off;
t_lick_ = lick_time(logical(lick_sensor));
for i = 2:length(t_lick_)
    if t_lick_(i) - t_lick_(i-1) < 0.005001
        t_lick_(i-1) = 0;
    end
end
t_lick_ = t_lick_(t_lick_ > 0);
diff = sum(abs(t_lick - t_lick_));
figure(10);
plot(t_lick);
hold on;
plot(t_lick_);
hold off;
xlabel('Lick #');
ylabel('Time (s)');
fig_id = fig_id + 1;

%% Classify lick in patch
t_patch = reshape(t_switch * dt_s, 2, [])';

% Test patch 1
t_lick_i = t_lick(in_interval(t_lick, t_patch(1, :)));
t_target_i = t_target(in_interval(t_target(:, 1), t_patch(1, :)), :); 
lick_result_i = zeros(length(t_lick_i), 1);

% Iterate through each lick to classify as:
% - Hit: lick during target (1)
% - FA: lick during no target (2)
% - N/A: within 1.0 second of previously classified lick (0)
pause_thresh = 1.0; % min duration between licks to count as separate lick train
num_hit = 0;
num_fa = 0;
current_target = 1;

is_hit = in_interval(t_lick_i(1), t_target_i(current_target, :));
if is_hit
    lick_result_i(1) = 1;
    num_hit = num_hit + 1;
    current_target = current_target + 1;
else
    lick_result_i(1) = 2;
    num_fa = num_fa + 1;
end
for i = 2:length(t_lick_i)
    % Determine if lick in same target as previous
    if current_target > 1
        in_prev_target = in_interval(t_lick_i(i), t_target_i(current_target-1, :));
    else
        in_prev_target = false;
    end

    % Classify lick only if 1) beginning of new lick train and 2) did not
    % occur during same target as previous lick
    if (t_lick_i(i) - t_lick_i(i-1) > pause_thresh) && ~in_prev_target
        is_hit = in_interval(t_lick_i(i), t_target_i(current_target, :));
        if is_hit
            lick_result_i(i) = 1;
            num_hit = num_hit + 1;
            current_target = current_target + 1;
        else
            lick_result_i(i) = 2;
            num_fa = num_fa + 1;
        end
    end
end

%% Plot lick raster on spectrogram
figure(fig_id);
clf(fig_id);
colormap(bone);
surf(t2, f2/1000, log10(abs(s2).^2), 'EdgeColor', 'none'); % plot estimate of PSD
axis xy; axis tight; view(0,90); % make similar to spectrogram() fn
xlabel('Time (s)');
ylabel('Frequency (kHz)');
colorbar;
hold all;
plot(xlim, [f_target/1000, f_target/1000], 'Color', 'black', 'LineStyle', '--');

y = f_target/1000; % height of raster plot
sz = diff(ylim)*0.05; % size of ticks
c = [[0.4660*0.75, 0.6740*0.75, 0.1880*0.75]; % hit
     [0.6350, 0.0780, 0.1840]; % fa
     [0.25, 0.25, 0.25]]; % n/a
p1 = raster_plot(t_lick_i(lick_result_i == 1), y, sz, c(1, :));
p2 = raster_plot(t_lick_i(lick_result_i == 2), y, sz, c(2, :));
p3 = raster_plot(t_lick_i(lick_result_i == 0), y, sz, c(3, :));
ps = [p1 p2 p3];
names = {'hit (lick)', 'fa (lick)', 'n/a (lick)'};
for i = 1:length(ps)
    if ~isempty(ps(i))
        legend(ps(i), names{i});
    end
end
hold off;
%fig_id = fig_id + 1;

% Calculate in-patch hit and fa rates
target_duration = pe.load_var('Settings', false);
target_duration = str2double(target_duration.Property.SoundConfigurationTargetSoundConfigTargetDurationsec);
num_targets = size(t_target_i, 1);
hit_rate_i = num_hit / num_targets;
num_cr = (diff(t_patch(1, :)) - (num_fa * target_duration)) / target_duration;
fa_rate_i = num_fa / (num_fa + num_cr);
if verbose
    % display hit and fa rate
    hit_rate_i
    fa_rate_i
end

%% Compare lick results with motor and trial data
% Find motor activation timestamps
motor = pe.load_var('UntitledMotor[1-4]', true);
dt_motor = motor.Property.wf_increment;
window_motor = max(round(window / dt_motor), 1);
motor = motor.Data(window_motor(1):end-window_motor(2));
motor_thresh = 1.0;
t_motor = find( (motor > motor_thresh) - circshift((motor > motor_thresh), 1) == 1);
t_motor = t_motor * dt_motor;

%% Classify motor in patch

t_motor_i = t_motor(in_interval(t_motor, t_patch(1, :)));
motor_result_i = zeros(length(t_motor_i), 1);
%% Iterate through each reward pump

% Iterate through each reward pump to classify as:
% - Hit: pump during (or immediately after) target (1)
% - Catch: pump not during (or immediately after) target (2)
num_hit = 0;
num_catch = 0;
current_target = 1;
pad = [0.0 1.0]; % time before/after target to associate motor with target

for i = 1:length(t_motor_i)
    % Classify motor based on proximity to target interval
    is_hit = in_interval(t_motor_i(i), t_target_i(current_target, :) + pad);
    if is_hit
        motor_result_i(i) = 1;
        num_hit = num_hit + 1;
        current_target = current_target + 1;
    else
        motor_result_i(i) = 2;
        num_catch = num_catch + 1;
    end
end

%% Plot motor raster on spectrogram
figure(fig_id);
y = f_target/1000 - diff(ylim)*0.10; % height of raster plot
sz = diff(ylim)*0.05; % size of ticks
c = [[0.4660*0.75, 0.6740*0.75, 0.1880*0.75]; % hit
     [0.6350, 0.0780, 0.1840]]; % catch
hold all;
p1 = raster_plot(t_motor_i(motor_result_i == 1), y, sz, c(1, :));
p2 = raster_plot(t_motor_i(motor_result_i == 2), y, sz, c(2, :));
ps = [p1 p2];
names = {'hit (motor)', 'catch (motor)'};
for i = 1:length(ps)
    if ~isempty(ps(i))
        legend(ps(i), names{i});
    end
end
hold off;
fig_id = fig_id + 1;

% Calculate in-patch hit rate and compare with previous calculation
hit_rate_i_motor = num_hit / num_targets;
if verbose
    hit_rate_i_motor
end
if hit_rate_i_motor ~= hit_rate_i
    % display hit and fa rate
    msg = 'Hit rates are not equal: %d (lick) vs. %d (motor)';
    id = 'PatchAnalysis:CalculationMismatch';
    warning(id, msg, hit_rate_i, hit_rate_i_motor);
end

%% Functions
function is_in_interval = in_interval(t, varargin)
    % Returns boolean of whether or not time t is in given interval %
    
    % Handle args
    if (length(varargin) == 1)
        t1 = varargin{1}(1);
        t2 = varargin{1}(2);
    elseif (length(varargin) == 2)
        t1 = varargin{1};
        t2 = varargin{2};
    else
        error('Number of input argument must be one or two.');
    end
    
    % t is in interval if t1 < t < t2
    after_t1 = t > t1;
    before_t2 = t < t2;
    is_in_interval = ((after_t1 + before_t2) == 2);
end

function p = raster_plot(xs, y, sz, c)
    ax = gca;
    co = ax.ColorOrderIndex;
    p = [];
    for i = 1:length(xs)
        ax.ColorOrderIndex = co;
        p = line([xs(i), xs(i)], [y - sz/2, y + sz/2], 'LineWidth', 0.5, 'Color', c);
    end
    ax.ColorOrderIndex = co + 1;
end