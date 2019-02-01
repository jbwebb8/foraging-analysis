%% Mostly a compilation of scripts for debugging experiments in more detail
% Load experiment
filename = 'j1z3_d25_2018_05_18_10_25_04.mat';
pe = PatchExperiment(filename);

%% Debug patch activity: plot wheel speed, position, stop points, and sound power
stop_thresh = 0.1;
run_thresh = 1.5;

wheel_speed = pe.wheel_speed(1000);
t_stop = pe.get_stop_times(stop_thresh, run_thresh);
[t_p_, t_t_, in_patch] = pe.get_patch_times();
x_t = pe.linear_position(1000);
P = pe.sound_power(1000, 500);

figure(1);
clf;
hold on;
yyaxis left;
plot(wheel_speed);
scatter(t_stop, ones(length(t_stop), 1)*0.05);
plot(P * max(wheel_speed) / max(P));
ylim([0 max(wheel_speed)*1.1])
yyaxis right;
plot(x_t);
fig = area(max(x_t) * in_patch);
fig.FaceAlpha = 0.2;
fig.FaceColor = 'blue';
fig.EdgeColor = 'none';
ylim([0 max(x_t)*1.1])
hold off;

%% Variant of debugging plot above
figure(1);
clf;
hold on;
yyaxis left;
patch_bounds = pe.get_patch_boundaries();
for i = 1:length(patch_bounds)
    x = 0;
    y = patch_bounds(1, i);
    w = length(x_t);
    h = pe.d_patch;
    rect = rectangle('Position', [x, y, w, h]);
    rect.FaceColor = [0.75 0.75 0.75];
    rect.EdgeColor = 'none';
end
plot(x_t);
scatter(t_stop, x_t(t_stop));
ylim([0 max(x_t)*1.1])
ylabel('linear position (cm)');

yyaxis right;
plot(wheel_speed, 'Linestyle', ':');
ylim([0 max(wheel_speed)*3])
ylabel('wheel speed (cm/s)');

xlabel('time (ms)');
hold off;