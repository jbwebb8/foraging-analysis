%% Extract patch data
% Arguments:
% - filelist: txt file containing file names of all training days of one 
%   mouse, in numerical order
% - plot_data: plot file data (e.g. power vs. time) while processing
filelist = importdata('G:\My Drive\Projects\foraging\data\matlist.txt');
plot_data = false;
t_p = zeros(size(filelist, 1), 2);
t_t = zeros(size(filelist, 1), 2);
r_p = zeros(size(filelist, 1), 2);

for i = 1:size(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    pe = PatchExperiment(filename);
    
    % Get patch statistics
    [t_p_i, t_t_i, in_patch_i] = pe.get_patch_times();
    r_p_i = pe.get_patch_rewards();
    t_p(i, :) = [mean(t_p_i), std(t_p_i)];
    t_t(i, :) = [mean(t_t_i), std(t_t_i)];
    r_p(i, :) = [mean(r_p_i), std(r_p_i)];
end

% Save data
[start_idx, end_idx] = regexp(filename, 'j[0-9]+[a-z][0-9]+_');
%new_filename = [filename(start_idx:end_idx), 'patch_data.mat'];
base_name = filename(1:end_idx);
new_filename = [base_name, 'patch_data.mat'];
fprintf('Saving to %s\n', new_filename);
save(new_filename, 't_p', 't_t', 'r_p', 'filelist');

%% Plot patch statistics
if ~(exist('t_p', 'var') && exist('t_t', 'var') && exist('r_p', 'var'))
    load(new_filename);
end

% Plot patch residence times
figure(1);
fig = errorbar(t_p(:, 1) / 1000, t_p(:, 2) / 1000);
title('Patch Residence Time');
xlabel('Training Day');
ylabel('Time (s)');
xlim([0 length(t_p)+1]);
saveas(fig, [base_name, 't_p.png']);

% Plot travel times
figure(2);
fig = errorbar(t_t(:, 1) / 1000, t_t(:, 2) / 1000);
title('Travel Time Between Patches');
xlabel('Training Day');
ylabel('Time (s)');
xlim([0 length(t_t)+1]);
saveas(fig, [base_name, 't_t.png']);

% Plot average patch rewards
figure(3);
fig = errorbar(r_p(:, 1), r_p(:, 2));
title('Average Cumulative Reward per Patch');
xlabel('Training Day');
ylabel('Reward (uL)');
xlim([0 length(r_p)+1]);
saveas(fig, [base_name, 'r_p.png']);

%% Object-based (above needs conversion)
% Plot distribution of stopping points
d_next_patch = pe.stopping_distances();
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