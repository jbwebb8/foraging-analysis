
%% Load data file
filename = 'D:\james\experiment_data\4-24-18\j1z1_d8_2018_04_24_09_30_34.mat';
load(filename);

%% Convert sound waveform to patch distances
% Raw audio waveform
s = UntitledSound1.Data;
dt_s = UntitledSound1.Property.wf_increment;

function P = 
% Convert to ms time scales
dt = 1e-3; % desired unit time
t_total = length(s) * dt_s / dt; % total time in ms

% P = rms(s) (averaged over bins of length t)
bin_len = 1000; % bin length
s_ = s(1:numel(s)-rem(numel(s), bin_len));
P = (sum(reshape(s_, bin_len, []).^2, 1) ./ bin_len).^0.5; % reshape is column-first operation
P = P(:); % vectorize P
sample_ratio = fix(t_total / length(P));
P = interp(P, sample_ratio);
figure(1);
plot(P);

%%
% Define patch and inter-patch time stamps
thresh = 0.008;
in_patch = (P > thresh); % true if in patch at time t
figure(1);
hold on;
fig = area(max(P) * in_patch);
fig.FaceAlpha = 0.2;
fig.FaceColor = 'blue';
fig.EdgeColor = 'none';
hold off;

%%
% Find minimum patch residence and travel times possible
v_max = max(UntitledWheelSpeed.Data) * 100; % cm/s
d_patch = 15; % cm; future data files should contain parameter
t_p_min = (d_patch / v_max) / dt; % ms
d_interpatch = 60; % cm
t_t_min = (d_interpatch / v_max) / dt; % ms

t_switch = find((in_patch + circshift(in_patch, -1)) == 1);
t_current = t_switch(1);
num_switches = 1;
in_patch_t = false;
true_switch = zeros(length(t_switch), 1);
for i = 1:length(t_switch)
    if i > 1
        t_current = t_current + (t_switch(i) - t_switch(i-1));
        num_switches = num_switches + 1;
    end
    if in_patch_t
        t_min = t_p_min;
    else
        t_min = t_t_min;
    end
    if (t_current >= t_min) && (rem(num_switches, 2) == 1)
        true_switch(i) = 1;
        t_current = 0;
        num_switches = 0;
    end   
end
t_switch = t_switch(logical(true_switch));
figure(1);
hold on;
scatter(t_switch, thresh * ones(size(t_switch)));
hold off;

%%
% If end in middle of patch, drop last patch. Otherwise, drop last travel
% time.
if rem(size(t_switch, 1), 2) == 1
    end_in_patch = true;
    t_switch = t_switch(1:length(t_switch-1));
else
    end_in_patch = false;
end

t_p = reshape(t_switch, 2, [])';
t_p = t_p(:, 2) - t_p(:, 1);

% Get travel times between patches by inference
t_t = circshift(t_switch, -1);
if end_in_patch
    t_t(length(t_t)) = t_switch(length(t_switch)); % recover last entry point
else
    t_t = t_t(1:length(t_t)-2); % drop first and last inter-patch
end
t_t = reshape(t_t, 2, [])';
t_t = t_t(:, 2) - t_t(:, 1);

figure(2);
semilogy(t_p);
figure(3);
semilogy(t_t);

%% Get reward data
% Another surrogate for patch residence is the reward: it decreases with
% each trial in the patch, then jumps up to the initial value at the first
% trial in the next patch. This may not match the total number of patches,
% however, since a trial may not occur if the mouse runs through a patch
% too quickly.

r_t = UntitledRewarduL.Data;
patch_idx = find(circshift(r_t, -1) >= r_t);
patch_idx = [0; patch_idx]; % handle end case

r_p = zeros(size(patch_idx)-1);
for i = 1:size(patch_idx)-1
    r_p(i) = sum(r_t(patch_idx(i)+1:patch_idx(i+1)));
end

figure(4);
plot(r_p);

%% Save data
filename = 'j1z1_d8_patch_data.mat'
save(filename, 't_p', 't_t', 'r_p');