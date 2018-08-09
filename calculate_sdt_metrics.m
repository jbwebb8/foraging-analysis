function [hit_rate, fa_rate, d_prime, c] = calculate_sdt_metrics(results, min_trials)
    % Calculates hit rate, false alarm rate, d`, and c given trial results
    % Args:
    % - results: vector of result of each trial, such that:
    %   0 Hit
    %   1 Miss
    %   2 FA
    %   3 Catch trial
    %   4 NA
    % - min_trials: minimum number of trials to calculate valid metric
    % Returns:
    % - hit_rate, fa_rate, d_prime, c: see SDT for definitions
    % 
    % Calculations:
    % Hit rate is sum(results == 0) / sum((results == 0) + (results == 1))
    % FA rate can be naively calculated as sum(results == 4) / sum(results > 1);
    % however, a better way is to sum the total time of not licking while no 
    % signal is present during each trial type (i.e. continuous assessment of
    % whether or not signal is present):
    % - Hit/miss: time to target start
    % - FA: time to FA
    % - Catch: duration of trial
    % Divide this total time by the duration of the target playing (i.e. how 
    % many time blocks), and then divide the total number of FAs by this number
    
    
    % Calculate naive fa rate = (# licks | no target) / (# no target)
    % TODO: this is slightly off, calculate more accurate value as above
    num_ns_trials = sum(results > 1);
    if num_ns_trials >= min_trials
        fa_rate = sum(results == 4) / num_ns_trials;
    else
        fa_rate = 0.0;
    end

    % Double number of trials, (2m-1)/(2m)
    % Calculate hit rate = (# licks | target) / (# target)
    num_s_trials = sum((results == 0) + (results == 1));
    if num_s_trials >= min_trials
        hit_rate = sum(results == 0) / num_s_trials;
    else
        hit_rate = fa_rate;
    end

    % Calculate d` and c
    if ~(isnan(hit_rate) || isnan(fa_rate))
        % Calculate z-scores for hit rate and fa rate. Assume they are
        % sampled from a cumulative Gaussian distribution ~ N(0, 1).
        mu = 0.0;
        sigma = 1.0;
        pd = makedist('Normal',mu,sigma);
        hit_rate_in = max(min(hit_rate, 0.99), 0.01); % avoids z = +/-inf
        fa_rate_in = max(min(fa_rate, 0.99), 0.01); % avoids z = +/-inf
        z_hit_rate = icdf(pd, hit_rate_in) / sigma;
        z_fa_rate = icdf(pd, fa_rate_in) / sigma;

        % Calculate d` = z(hit_rate) - z(fa_rate)
        d_prime = z_hit_rate - z_fa_rate;

        % Calculate c = -(z(hit_rate) + z(fa_rate)) / 2
        c = -(z_hit_rate + z_fa_rate) / 2; % note negative sign
    else
        % Set values to be undefined
        d_prime = NaN;
        c = NaN;
    end
end