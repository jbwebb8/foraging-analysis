function [t_p, t_t, r_p] = get_patch_data(filename, plot_data)
    % Load data files
    function struct = load_var(var_name, use_regex)
        if use_regex    
            struct = load(filename, '-regexp', var_name); % struct.var_name
        else
            struct = load(filename, var_name); % struct.var_name
        end
        fieldname = fieldnames(struct); % {'var_name'}
        fieldname = fieldname{1}; % 'var_name'
        struct = getfield(struct, fieldname); % var_name
    end

    % Convert sound waveform to patch distances
    % Raw audio waveform
    struct = load_var('UntitledSound[1-4]', true);
    s = struct.Data;
    dt_s = struct.Property.wf_increment;

    % Convert to ms time scales
    dt = 1e-3; % desired unit time
    t_total = length(s) * dt_s / dt; % total time in ms

    % P = rms(s) (averaged over bins of length t)
    bin_len = 1000; % bin length
    s_ = s(1:numel(s)-rem(numel(s), bin_len));
    P = (sum(reshape(s_, bin_len, []).^2, 1) ./ bin_len).^0.5; % reshape is column-first operation
    P = interp1(1:length(P), P, linspace(1, length(P), t_total));
    %sample_ratio = fix(t_total / length(P));
    %P = interp(P, sample_ratio); % interpolate to ms
    P = P(:); % vectorize P
    filter_len = 500; % length (ms) of moving average filter
    P = filter(1/filter_len * ones(filter_len, 1), 1, P); % smooth P

    % Define patch and inter-patch time stamps
    thresh = 0.008;
    in_patch = (P > thresh); % true if in patch at time t

    % Find minimum patch residence and travel times possible
    struct = load_var('UntitledWheelSpeed', false);
    v_max = max(struct.Data) * 100; % cm/s
    d_patch = 15; % cm; future data files should contain parameter
    t_p_min = (d_patch / v_max) / dt; % ms
    d_interpatch = 60; % cm
    t_t_min = (d_interpatch / v_max) / dt; % ms

    % Find timestamps associated with patch entry/exit
    % 1) Over/under threshold crossing --> patch entry/exit
    t_switch = find((in_patch + circshift(in_patch, -1)) == 1);

    % 2) Ensure that patch/interpatch times are feasible
    t_current = t_switch(1); % current length of patch/interpatch in ms
    num_switches = 1; % number of threshold crossings since last entry/exit
    in_patch_t = false; % currently in patch
    true_switch = zeros(length(t_switch), 1); % feasible thresh crossings 
    for i = 1:length(t_switch)
        % Add duration of next switch
        if i > 1
            t_current = t_current + (t_switch(i) - t_switch(i-1));
            num_switches = num_switches + 1;
        end

        % Get minimum feasible time between switches
        if in_patch_t
            t_min = t_p_min;
        else
            t_min = t_t_min;
        end

        % Switch corresponds to actual entry/exit if time is feasible
        % and number of thresh crossings is even
        if (t_current >= t_min) && (rem(num_switches, 2) == 1)
            true_switch(i) = 1;
            t_current = 0;
            num_switches = 0;
        end  
    end
    
    % Update t_switch to checked version
    t_switch = t_switch(logical(true_switch));

    % If end in middle of patch, drop last patch. 
    % Otherwise, drop last travel time.
    if rem(size(t_switch, 1), 2) == 1
        end_in_patch = true;
        t_switch = t_switch(1:length(t_switch)-1);
    else
        end_in_patch = false;
    end
    
    % patch residence time = t_exit - t_enter
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

    % Get reward data
    % Another surrogate for patch residence is the reward: it decreases with
    % each trial in the patch, then jumps up to the initial value at the first
    % trial in the next patch. This may not match the total number of patches,
    % however, since a trial may not occur if the mouse runs through a patch
    % too quickly.
    struct = load_var('UntitledRewarduL', false);
    r_t = struct.Data;
    patch_idx = find(circshift(r_t, -1) >= r_t);
    patch_idx = [0; patch_idx]; % handle end case

    r_p = zeros(size(patch_idx)-1);
    for i = 1:size(patch_idx)-1
        r_p(i) = sum(r_t(patch_idx(i)+1:patch_idx(i+1)));
    end

    % Plot data if specified
    if plot_data
        % Figure 1: power vs. time (ms); patches are shaded and filtered
        % points of patch enter/exit marked with circles
        figure(1);
        plot(P);
        hold on;
        fig = area(max(P) * in_patch);
        fig.FaceAlpha = 0.2;
        fig.FaceColor = 'blue';
        fig.EdgeColor = 'none';
        scatter(t_switch, thresh * ones(size(t_switch)));
        hold off;

        % Figure 2: semilog plot of patch residence vs. patch number
        figure(2);
        semilogy(t_p);

        % Figure 3: semilog plot of travel time vs. interpatch number
        figure(3);
        semilogy(t_t);

        % Figure 4: total reward vs. patch number
        figure(4);
        plot(r_p);
    end
end