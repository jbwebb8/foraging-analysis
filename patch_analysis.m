%% Set parameters and create filelist
% Arguments:
% - filelist: txt file containing file names of all training days of one 
%   mouse, in numerical order
% - plot_data: plot file data (e.g. power vs. time) while processing
% - use_sound: use sound waveform for defining patches if position not available
filelist = 'matlist.txt';
plot_data = false;
use_sound = false;
stop_thresh = 0.1;
run_thresh = 0.5;

% Sort filelist and remove irrelevant filenames
[filelist, training_days] = sort_training_files(filelist);
start_idx = regexp(filelist{1}, 'j[0-9][a-z][0-9]_d');
mouse_id = filelist{1}(start_idx:start_idx+3);

%% Extract patch data
% Set data placeholders for each experiment
t_p = cell(size(filelist, 1), 1); % residence time
t_t = cell(size(filelist, 1), 1); % travel time
r_p = cell(size(filelist, 1), 1); % total reward per patch
d_next_patch = cell(size(filelist, 1), 1); % distance to next patch from stopping points
d_config = zeros(size(filelist, 1), 2); % [d_patch, d_interpatch]
keep_idx = ones(size(filelist, 1), 1); % files to include in analysis

for i = 1:size(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    pe = PatchExperiment(filename);
    
    if ~pe.use_sound
        % Get patch statistics
        [t_p{i}, t_t{i}, in_patch_i] = pe.get_patch_times();
        r_p{i} = pe.get_patch_rewards();
        d_next_patch{i} = pe.stopping_distances(stop_thresh, run_thresh);
        d_config(i, :) = [pe.d_patch, pe.d_interpatch];
    else
        % Skip file if does not contain position data
        fprintf('%s does not contain position data. Skipping file.\n', filename);
        keep_idx(i) = 0;
    end
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

%% Save data
[start_idx, end_idx] = regexp(filename, 'j[0-9]+[a-z][0-9]+_');
%new_filename = [filename(start_idx:end_idx), 'patch_data.mat'];
base_name = filename(1:end_idx);
new_filename = [base_name, 'patch_data.mat'];
fprintf('Saving to %s\n', new_filename);
save(new_filename, 't_p', 't_t', 'r_p', 'd_next_patch', 'd_config', 'filelist');

%% Plot patch statistics
% Load file if not already loaded
if ~(exist('t_p', 'var') && exist('t_t', 'var') && exist('r_p', 'var'))
    load(new_filename);
end
save_fig = true;

% Get training days
if ~exist('training_days', 'var')
    training_days = zeros(length(filelist), 1);
    for i = 1:length(filelist)
        filename = filelist{i};
        [start_idx, end_idx] = regexp(filename, '_d[0-9]+_');
        training_days(i) = str2double(filename(start_idx+2:end_idx-1));
    end
end

% Plot patch residence times
fig1 = figure(1);
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

% Plot travel times
fig2 = figure(2);
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

% Plot average patch rewards
fig3 = figure(3);
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

% Plot distribution of stopping points
% histogram plots
%{
figure(4);
edges_interpatch = [-pe.d_interpatch/2:5:0, pe.d_patch:5:pe.d_interpatch/2];
edges_patch = 0:5:pe.d_patch;
histogram(d_next_patch(d_next_patch<0 | d_next_patch>pe.d_patch), ...
          edges_interpatch);
hold on;
histogram(d_next_patch(d_next_patch>=0 & d_next_patch<=pe.d_patch), ...
          edges_patch); 
[f, xi] = ksdensity(d_next_patch);
plot(xi, f*1/max(f));
hold off;
figure(5);
[N_1, edges_1] = histcounts(d_next_patch(d_next_patch>=0 & d_next_patch<=pe.d_patch), ...
          edges_patch);
w_1 = (edges_1(2) - edges_1(1)) / 2;
edges_1 = edges_1(1:end-1) + w_1;
hold on;
[N_2, edges_2] = histcounts(d_next_patch(d_next_patch<0 | d_next_patch>pe.d_patch), ...
          edges_interpatch);
w_2 = (edges_2(2) - edges_2(1)) / 2;
edges_2 = edges_2(1:end-1) + w_2;
edges_2 = edges_2(edges_2<0 | edges_2>pe.d_patch);
edges = [edges_1, edges_2];
plot(edges, [N_1, zeros(1, length(edges_2))]);
plot(edges, [zeros(1, length(edges_1)), N_2]);
hold off;
%}

fig6 = figure(6);
clf(fig6);
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

%% Save figures if specified
if save_fig
    saveas(fig1, [base_name, 't_p']);
    saveas(fig2, [base_name, 't_t']);
    saveas(fig3, [base_name, 'r_p']);
    saveas(fig6, [base_name, 'd_stop']);
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
            if strcmp(ME.identifier, '')
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
    fig = figure(j);
    clf(fig);
    plot(vars{j}(:, 1), vars{j}(:, 2));
    %ylim([-2 2]);
    ylabel(var_names{j});
    xlabel('Training Day');
end

%% Analyze relationship between behavior and time
% Does performance worsen with time and/or reward accrued? In order to
% answer, let's plot a sliding window of the average hit rate, etc. over
% the last n trials vs. total reward accrued.

% Trial results:
% 0 Hit
% 1 Miss
% 2 ?
% 3 ? (happens only on catch trials)
% 4 FA?
% Hit rate is sum(results == 0) / sum((results == 0) + (results == 1))
% Unclear how FA rate is calculated: sum(results == 4) / sum(results > 1)
% is close but not exact. For now, we'll just use this measure.

% Params
m = 30; % length of sliding window (# trials)
n = 4; % minimum # of trials to calculate metric

% Set up matrices
hit_rate = cell(length(filelist), 1);
fa_rate = cell(length(filelist), 1);
d_prime = cell(length(filelist), 1);
c = cell(length(filelist), 1);
r_cum = cell(length(filelist), 1);

for i = 1:length(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    pe = PatchExperiment(filename);
    
    % Get cumulative rewards over trials
    struct = pe.load_var('UntitledRewarduL', false);
    rewards = struct.Data;
    r_cum{i} = cumsum(rewards);
    
    % Slide along trial outcomes with window of size m
    struct = pe.load_var('UntitledTrialResult', false);
    results = struct.Data;
    num_trials = length(results);
    hit_rate{i} = zeros(num_trials, 1);
    fa_rate{i} = zeros(num_trials, 1);
    d_prime{i} = zeros(num_trials, 1);
    c{i} = zeros(num_trials, 1);
    for j = 1:num_trials
        % Get most recent m results
        results_m = results(max(1, j-m+1):j);
        
        % Calculate fa rate = (# licks | no target) / (# no target)
        % TODO: this is slightly off, not sure why
        % New way: sum total time of no licking while no signal during
        % each trial type
        % - Hit/miss: time to target start
        % - FA: time to FA
        % - Catch: duration of trial
        % Divide this total time by the duration of the target(i.e. how many
        % time blocks), and then divide the total number of FAs by this number
        num_ns_trials = sum(results_m > 1);
        if num_ns_trials >= n
            fa_rate{i}(j) = sum(results_m == 4) / num_ns_trials;
        else
            fa_rate{i}(j) = 0.0;
        end
        
        % Double number of trials, (2m-1)/(2m)
        % Calculate hit rate = (# licks | target) / (# target)
        num_s_trials = sum((results_m == 0) + (results_m == 1));
        if num_s_trials >= n
            hit_rate{i}(j) = sum(results_m == 0) / num_s_trials;
        else
            hit_rate{i}(j) = fa_rate{i}(j);
        end
        
        % Calculate d` and c
        if ~(isnan(hit_rate{i}(j)) || isnan(fa_rate{i}(j)))
            % Calculate z-scores for hit rate and fa rate. Assume they are
            % sampled from a cumulative Gaussian distribution ~ N(0, 1).
            mu = 0.0;
            sigma = 1.0;
            pd = makedist('Normal',mu,sigma);
            hit_rate_in = max(min(hit_rate{i}(j), 0.99), 0.01); % avoids z = +/-inf
            fa_rate_in = max(min(fa_rate{i}(j), 0.99), 0.01); % avoids z = +/-inf
            z_hit_rate = icdf(pd, hit_rate_in) / sigma;
            z_fa_rate = icdf(pd, fa_rate_in) / sigma;

            % Calculate d` = z(hit_rate) - z(fa_rate)
            d_prime{i}(j) = z_hit_rate - z_fa_rate;

            % Calculate c = (z(hit_rate) + z(fa_rate)) / 2
            c{i}(j) = -(z_hit_rate + z_fa_rate) / 2; % no negative sign
        else
            % Set values to be undefined
            d_prime{i}(j) = NaN;
            c{i}(j) = NaN;
        end
    end
end

% Plot results
for i = 1:4
    clf(figure(i));
    hold on;
end

for i = 1:length(filelist)
    figure(1);
    plot(hit_rate{i});
    
    figure(2);
    plot(fa_rate{i});
    
    figure(3);
    plot(d_prime{i});
    
    figure(4);
    plot(c{i});
end
hold off;

y_labels = {'Hit Rate', 'FA Rate', 'd`', 'c'};
for i = 1:4
    figure(i);
    title(mouse_id);
    xlabel('Trial #');
    ylabel(y_labels{i});
    legend("Day " + string(training_days));
end
%% Analyze velocity traces
v = pe.wheel_speed();