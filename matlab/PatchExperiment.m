classdef PatchExperiment < handle
    properties
        
        filename % data file (.mat)
        use_sound % true if position info not available
        fig_id % avoids overwriting figures
        verbose % plot intermediate figures, output intermediate values, etc.
        
        t_total % total time of experiment in ms
        dt % incremental time step (default ms)
        t_active % active time window [t_start, t_end]
        
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
            % Constructor %
             
            % Set global settings
            obj.filename = filename;
            obj.fig_id = 1;
            obj.verbose = false;
            
            % Get patch and interpatch distances. Two different naming conventions.
            try
                struct = obj.load_var('Settings', false);
                obj.d_patch = str2double(struct.Property.SoundConfigurationRunConfigPatchLengthcm);
                obj.d_interpatch = str2double(struct.Property.SoundConfigurationRunConfigInterPatchDistcm);
            catch
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
            end
            
            % Get total time in ms from wheel sampling (much cheaper than
            % loading sound)
            struct = obj.load_var('UntitledWheelTime', false);
            t_total = struct.Data(end); % last sample time in sec
            obj.dt = 1e-3; % desired unit time
            obj.t_total = int32(t_total / obj.dt); % total time in ms
            
        end
        
        function set_t_total_from_sound(obj)
        
            % Get total time in ms from sound (known sampling rate)
            struct = obj.load_var('UntitledSound[1-4]', true);
            s = struct.Data;
            dt_s = struct.Property.wf_increment;
            obj.t_total = int32(length(s) * dt_s / obj.dt); % total time in ms
        
        end
        
        function s = get_spectrogram_variance(obj, fft_pts, lp_pass, lp_stop)
            %%% Finish incorporating code from patch_analysis to PatchExperiment %%%
            
            % Load sound data
            struct = obj.load_var('UntitledSound[1-4]', true);
            sound_wf = struct.Data;
            dt_wf = struct.Property.wf_increment;

            % Get spectrogram of sound waveform
            s = spectrogram(sound_wf, fft_pts); % complex frequency content; shape=[freq, time]
            s_abs = abs(s); % complex magnitude; shape=[freq, time]

            % Create filter
            d = fdesign.lowpass('Fp,Fst,Ap,Ast', ... % format
                                2*lp_pass*dt_wf*fft_pts, ... % Fp
                                2*lp_stop*dt_wf*fft_pts, ... % Fst
                                1, 60); % Ap, Ast
            %designmethods(d) % lists design choices
            Hd = design(d, 'butter'); % Butterworth IIR filter
            if obj.verbose
                fvtool(Hd); % visualize filter
            end

            % Calculate variance in frequency domain at each time point
            if obj.verbose
                % Spectrogram of complex magnitudes
                figure(obj.fig_id);
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
                obj.fig_id = obj.fig_id + 1;
            else
                % One-line calculation
                s_var = median(abs(diff(medfilt1(s_abs,1),1)));
            end

            % Smooth median variance
            s_var_filt = filtfilt(Hd.sosMatrix, Hd.ScaleValues, s_var); % zero-phase filtering
            if obj.verbose
                % Settings
                nbins = 10000;

                % Histogram and plot of median variance
                figure(obj.fig_id);
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
                obj.fig_id = obj.fig_id + 1;
            end
            
        end
        
        function [t_switch, in_patch] = get_patch_times_from_sound(...
                obj, fft_pts, lp_pass, lp_stop, thresh)
            
            % Convert sound waveform to patch distances
            s_var_filt = obj.get_spectrogram_variance(fft_pts, lp_pass, lp_stop);

            % Define patch and inter-patch time stamps
            in_patch = (s_var_filt > thresh); % true if in patch at time t
            t_switch = find((in_patch + circshift(in_patch, -1)) == 1);

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
            if obj.verbose
                figure(obj.fig_id);
                plot(x_t);
                hold on;
                fig = area(max(wheel_speed) * in_patch);
                fig.FaceAlpha = 0.2;
                fig.FaceColor = 'blue';
                fig.EdgeColor = 'none';
                hold off;
                obj.fig_id = obj.fig_id + 1;
            end
            
        end
        
        function [t_p, t_t] = get_patch_times(obj, t_switch, in_patch)
            % Get time information about patches
            % - t_p: [num_patches, 1] array representing patch residence times
            % - t_t: [num_patches, 1] array representing interpatch start and end points
            
            % If start in patch, treat time=0 as first patch entry. If end 
            % in middle of patch, drop last patch. Note that this combination
            % always results in t_switch having an even number of components.
            start_in_patch = in_patch(1);
            end_in_patch = xor((rem(length(t_switch), 2) == 1), start_in_patch);
            if start_in_patch
                t_switch = [0; t_switch]; % insert zero at beginning for patch entry   
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
            
            % Set class variables
            obj.t_p = t_p;
            obj.t_t = t_t;
            obj.in_patch = in_patch;
            obj.end_in_patch = end_in_patch;
            obj.num_patches = length(t_p) + end_in_patch;
            
        end
        
        % TODO: update to work with zero reward decay
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
        
        % Window functions: These return values that are dependent on the
        % active time window, which is set by the helper function
        % set_time_window() below.
        % Note: these will have to wait until a link can be made between
        % the trial and the time.
        function d_prime(obj)
            % Return d` (sensitivity) for active time window.
            
            % Get hit rate in active time window
            
            % Get FA rate in active time window
            
            % Get z-transform of both rates
            
            % Return d`
            
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
            x_t = (theta_t / 360 * 60)'; % column vector
            
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
            wheel_speed = wheel_speed'; % column vector
        end
        
        % Helper functions: Miscellaneous helper functions.
        function struct = load_var(obj, var_name, use_regex)
            % Load data files
             
            if use_regex    
                struct = load(obj.filename, '-regexp', var_name); % struct.var_name
            else
                struct = load(obj.filename, var_name); % struct.var_name
            end
            
            fieldname = fieldnames(struct); % {'var_name'}
            if ~isempty(fieldname)
                fieldname = fieldname{1}; % 'var_name'
                struct = getfield(struct, fieldname); % var_name
            else
                msg = 'Variable %s not found.';
                id = 'PatchExperiment:VarNotFound';
                error(id, msg, var_name);
            end
            
        end
        
        function set_time_window(obj, t_start, t_end)
            % Set current time window for dependent functions.
            % Args:
            % - t_start: start time in minutes
            % - t_end: end time in minutes
            
            t_start = t_start / obj.dt;
            t_end = t_end / obj.dt;
            obj.t_active = [t_start, t_end];
        
        end
        
        function clear_memory(obj)
            % Can set this function to clear either all large memory
            % object properties or specific list 
        end
    end
end
