classdef PatchExperiment < handle
    properties
        filename % data file (.mat)
        use_sound % true if position info not available
        t_total % total time of experiment in ms
        dt % incremental time step (default ms)
        d_patch % distance of a patch (cm)
        d_interpatch % distance between patches (cm)
        in_patch % true if in patch at time t (vector of length t_total)
        end_in_patch % true if experiment stopped in middle of patch
        num_patches % number of patches animal encountered (last patch dropped if end_in_patch true)
        t_p % 2D array of patch start and end times
        t_t % 2D array of travel start and end times
        r_p % vector of total reward received in each patch
    end
    
    methods
        function obj = PatchExperiment(filename)
            % Constructor
             
            % Set filename
            obj.filename = filename;
            
            % Determine if animal position recorded
            try
                struct = obj.load_var('UntitledAngularPosition', false);
                obj.use_sound = false;
            catch
                msg = ['Angular Position not found. Will use sound waveform ', ...
                       'to estimate patch locations in time.'];
                disp(msg);
                obj.use_sound = true;
            end
            
            % Get patch and interpatch distances
            try
                struct = obj.load_var('SoundConfiguration', false);
                obj.d_patch = str2double(struct.Property.RunConfigPatchLengthcm);
                obj.d_interpatch = str2double(struct.Property.RunConfigInterPatchDistcm);
            catch
                msg = 'Patch and interpatch distances not found. Using default values.';
                disp(msg);
                obj.d_patch = 15;
                obj.d_interpatch = 60;
            end
            
            % Get total time in ms from wheel sampling (much cheaper than
            % loading sound)
            struct = obj.load_var('UntitledWheelTime', false);
            t_total = struct.Data(end); % last sample time in sec
            obj.dt = 1e-3; % desired unit time
            obj.t_total = t_total / obj.dt; % total time in ms
            
        end
        
        function set_t_total_from_sound(obj)
        
            % Get total time in ms from sound (known sampling rate)
            struct = obj.load_var('UntitledSound[1-4]', true);
            s = struct.Data;
            dt_s = struct.Property.wf_increment;
            obj.t_total = int32(length(s) * dt_s / obj.dt); % total time in ms
        
        end
        
        function [t_switch, in_patch] = get_patch_times_from_sound(obj)
            % Note: due to errors in recording the sound waveform, this
            % method can be highly inaccurate. Position data preferred.
            
            % Convert sound waveform to patch distances
            bin_len = 1000;
            filter_len = 500;
            P = obj.sound_power(bin_len, filter_len);

            % Define patch and inter-patch time stamps
            thresh = 0.008;
            in_patch = (P > thresh); % true if in patch at time t

            % Find minimum patch residence and travel times possible
            struct = obj.load_var('UntitledWheelSpeed', false);
            v_max = max(struct.Data) * 100; % cm/s
            t_p_min = (obj.d_patch / v_max) / obj.dt; % ms
            t_t_min = (obj.d_interpatch / v_max) / obj.dt; % ms

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
                    in_patch_t = ~in_patch_t;
                end  
            end

            % Update t_switch to checked version
            t_switch = t_switch(logical(true_switch));
            
            % Update in_patch accordingly
            % Credit: https://bit.ly/2In6AzM
            in_patch = zeros(int32(obj.t_total), 1);
            if rem(size(t_switch, 1), 2) == 1
                t_switch_ = t_switch(1:end-1);
            else
                t_switch_ = t_switch;
            end
            t_switch_ = reshape(t_switch_', 2, [])';
            lo = t_switch_(:, 1); up = t_switch_(:, 2);
            n = cumsum([1; up(:) - lo(:) + 1]); % number of in_patch = true
            z = ones(n(end)-1,1); % initial vector
            z(n(1:end-1)) = [lo(1); lo(2:end)-up(1:end-1)]; % set skip points
            in_patch(cumsum(z)) = 1; % cumulative sum to get indices
            
        end
        
        function [t_switch, in_patch] = get_patch_times_from_position(obj)
            % Get time information about patches from position
            
            % Convert angular position to linear position
            filter_len = 1000;
            x_t = obj.linear_position(filter_len);
            patches = obj.get_patch_boundaries();
            
            % Animal in a patch when position within any of boundaries
            in_patch = (x_t >= patches(1, :) & x_t <= patches(2, :));
            in_patch = sum(in_patch, 2);
            t_switch = find((in_patch + circshift(in_patch, -1)) == 1);
            
            % Plot verification
            %figure(6);
            %plot(x_t);
            %figure(5);
            %hold on;
            %fig = area(max(wheel_speed) * in_patch);
            %fig.FaceAlpha = 0.2;
            %fig.FaceColor = 'blue';
            %fig.EdgeColor = 'none';
            %hold off;
            
        end
        
        function [t_p, t_t, in_patch] = get_patch_times(obj)
            % Get time information about patches from sound waveform
            % - t_p: [num_patches, 2] array representing patch start and end points
            % - t_t: [num_patches, 2] array representing interpatch start and end points
            % - in_patch: [t_total, 1] vector representing whether or not in patch
            if obj.use_sound
                [t_switch, in_patch] = get_patch_times_from_sound(obj);
            else
                [t_switch, in_patch] = get_patch_times_from_position(obj);
            end
            
            % If end in middle of patch, drop last patch. 
            % Otherwise, drop last travel time.
            end_in_patch = (rem(size(t_switch, 1), 2) == 1);
            if end_in_patch
                t_switch = t_switch(1:length(t_switch)-1);
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
            
            % Set class variables
            obj.t_p = t_p;
            obj.t_t = t_t;
            obj.in_patch = in_patch;
            obj.end_in_patch = end_in_patch;
            obj.num_patches = length(t_p) + end_in_patch;
            
        end
        
        function r_p = get_patch_rewards(obj)
            % Get total reward per reward
            
            % Load reward data
            struct = obj.load_var('UntitledRewarduL', false);
            r_t = struct.Data;
            
            % Find indices corresponding to separate patches
            patch_idx = find(circshift(r_t, -1) >= r_t);
            patch_idx = [0; patch_idx]; % handle end case

            % Sum rewards within each patch   
            % Currently all patches with zero rewards are at end of r_p,
            % not interlaced chronologically
            r_p = zeros(obj.num_patches, 1);
            for i = 1:length(patch_idx)-1
                r_p(i) = sum(r_t(patch_idx(i)+1:patch_idx(i+1)));
            end
            
            % Set object variable
            obj.r_p = r_p;
        end
        
        function plot_patch_data(obj)
            % Figure 1: power vs. time (ms); patches are shaded and filtered
            % points of patch enter/exit marked with circles
            figure(1);
            plot(P);
            hold on;
            fig = area(max(P) * obj.in_patch);
            fig.FaceAlpha = 0.2;
            fig.FaceColor = 'blue';
            fig.EdgeColor = 'none';
            scatter(t_switch, thresh * ones(size(t_switch)));
            hold off;

            % Figure 2: semilog plot of patch residence vs. patch number
            figure(2);
            semilogy(obj.t_p);

            % Figure 3: semilog plot of travel time vs. interpatch number
            figure(3);
            semilogy(obj.t_t);

            % Figure 4: total reward vs. patch number
            figure(4);
            plot(obj.r_p);
            
            % Need to make x_t an object property
            figure(5);
            plot(x_t);
            hold on;
            scatter(find(in_patch), in_patch(in_patch>0) .* x_t(in_patch>0));
            scatter(t_switch, ones(length(t_switch), 1) .* x_t(t_switch));
            hold off;
        end
        
        function t_stop = get_stop_times(obj, stop_thresh, run_thresh)
            % Determine when animal stopped running during experiment
            % Args:
            % - stop_thresh: if > speed, animal considered to switch from
            %                running to stopped (unit: cm/s)
            % - run_thresh: if < speed, animal considered to switch from
            %               stopped to running (unit: cm/s)
            
            
            % Get smoothed wheel speed
            filter_len = 1000;
            wheel_speed = obj.wheel_speed(filter_len);
            
            % Mark stopping point as first occurrence of wheel_speed <
            % stop_thresh after wheel_speed > run_thresh. This avoid
            % counting noisy oscillations around the threshold as multiple
            % stopping points.
            is_running = wheel_speed(1) > run_thresh; % initially running
            t_stop = zeros(obj.t_total, 1);
            for i = 1:obj.t_total
                if wheel_speed(i) > run_thresh
                    is_running = true;
                elseif (wheel_speed(i) < stop_thresh) && is_running
                    t_stop(i) = 1;
                    is_running = false;
                end
            end
            t_stop = find(t_stop); % Convert from bool vector to timestamps
            
        end
        
        function d_next_patch = stopping_distances(obj, stop_thresh, run_thresh)
            % stopping times
            filter_len = 1000;
            t_stop = obj.get_stop_times(stop_thresh, run_thresh);
            x_t = obj.linear_position(filter_len);
            x_stop = x_t(t_stop);
            patches = obj.get_patch_boundaries();

            % Determine distance to next patch for each stopping point.
            d_next_patch = patches(1, :) - x_stop; % distance to all patches
            [i, j] = find(d_next_patch > 0); % indices for positive distances
            [i_first, i_first_idx] = unique(i, 'first');
            j_first = j(i_first_idx);
            idx = (j_first-1) * size(d_next_patch, 1) + i_first; % indexing is column-first
            d_next_patch = d_next_patch(idx); % first positive distance

            % Shift distances to [-d_interpatch/2, d_patch, d_interpatch/2]
            d_next_patch = -d_next_patch + obj.d_interpatch; % flip patch to beginning
            d_next_patch(d_next_patch > obj.d_interpatch/2) = ...
                d_next_patch(d_next_patch > obj.d_interpatch/2) ...
                - (obj.d_interpatch + obj.d_patch); % flip late interpatch to beginning
            d_next_patch = d_next_patch + obj.d_patch; % shift 0 to start of patch
           
        end
        
        function patch_bounds = get_patch_boundaries(obj)
            % Create array of patch boundaries
            
            filter_len = 1000;
            x_t = obj.linear_position(filter_len);
            d1 = obj.d_interpatch; d2 = obj.d_patch; % for aesthetics
            max_num_patches = ceil(max(x_t) / (d1+d2));
            patch_bounds = zeros(2, max_num_patches);
            patch_bounds(1, :) = (d1:(d1+d2):max_num_patches*(d1+d2));
            patch_bounds(2, :) = ((d1+d2):(d1+d2):max_num_patches*(d1+d2)+d2);
            
        end
       
        % Intermediate values (these may need to be switched to object 
        % properties if loading multiple times is slow; currently this way
        % to preserve memory)
        function x_t = linear_position(obj, filter_len)
            % Convert angular position to linear position
            % Args:
            % - filter_len: length (ms) of moving average filter
            
            struct = obj.load_var('UntitledAngularPosition', false);
            theta_t = struct.Data;
            theta_t = filter(1/filter_len * ones(filter_len, 1), 1, theta_t); % smooth theta
            theta_t = interp1(1:length(theta_t), theta_t, ...
                              linspace(1, length(theta_t), obj.t_total)); % interpolate to t_total
            x_t = (theta_t / 360 * 60)';
            
        end
        
        function P = sound_power(obj, window_len, filter_len)
            % Convert sound waveform to smoothed power curve
            % Args:
            % - bin_len: length (ms) of bins for computing power
            % - filter_len: length (ms) of moving average filter
            
            % Raw audio waveform
            struct = obj.load_var('UntitledSound[1-4]', true);
            s = struct.Data;

            % P = rms(s) (averaged over bins of length t)
            %s_ = s(1:numel(s)-rem(numel(s), bin_len));
            %P = (sum(reshape(s_, bin_len, []).^2, 1) ./ bin_len).^0.5; % reshape is column-first operation
            kernel = ones(1, window_len) / window_len; % sliding filter to sum squares
            P = filter(kernel, 1, s.^2); % avg of sum of squares in window
            P = P .^ 0.5; % rms
            P = interp1(1:length(P), P, linspace(1, length(P), obj.t_total));
            P = P(:); % vectorize P
            P = filter(1/filter_len * ones(filter_len, 1), 1, P); % smooth P
            
        end
        
        function wheel_speed = wheel_speed(obj, filter_len)
            % Return smoothed wheel speed over time
            % Args:
            % - filter_len: length (ms) of moving average filter
            
            % Load wheel speed
            struct = load(obj.filename, 'UntitledWheelSpeed');
            wheel_speed = struct.UntitledWheelSpeed.Data * 100; % cm/s

            % Interpolate and smooth wheel speed
            wheel_speed = filter(1/filter_len * ones(filter_len, 1), 1, wheel_speed);
            wheel_speed = interp1(1:length(wheel_speed), wheel_speed, linspace(1, length(wheel_speed), obj.t_total));
        end
        
        % Helper functions
        function struct = load_var(obj, var_name, use_regex)
             % Load data files
             
            if use_regex    
                struct = load(obj.filename, '-regexp', var_name); % struct.var_name
            else
                struct = load(obj.filename, var_name); % struct.var_name
            end
            fieldname = fieldnames(struct); % {'var_name'}
            fieldname = fieldname{1}; % 'var_name'
            struct = getfield(struct, fieldname); % var_name
            
        end
        
        function clear_memory(obj)
            % Can set this function to clear either all large memory
            % object properties or specific list 
        end
    end
end
