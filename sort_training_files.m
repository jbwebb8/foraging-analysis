function [filelist, training_days] = sort_training_files(filenames)
    % Sorts and removes irrelevant filenames from list of matlab data files
    % 
    % Args:
    % - filenames: text file containing list of matlab data files (.mat)
    %
    % Returns:
    % - filelist: sorted and parsed list of filenames
    % - training_days: corresponding training day for each file
    
    % Get filenames from text file
    filelist = importdata(filenames);
    
    % Sort filelist
    training_days = zeros(length(filelist), 1);
    for i = 1:length(filelist)
        filename = filelist{i};
        [start_idx, end_idx] = regexp(filename, '_d[0-9]+_');
        training_days(i) = str2double(filename(start_idx+2:end_idx-1));
    end 
    [sorted_days, sorted_idx] = sort(training_days);
    
    % Remove files not containing training data (not containing '_d[0-9]+')
    nan_idx = find(isnan(sorted_days));
    if ~isempty(nan_idx)
        sorted_days = sorted_days(1:nan_idx-1);
        sorted_idx = sorted_idx(1:nan_idx-1);
    end
    
    % Return sorted filelist and training days
    filelist = filelist(sorted_idx);
    training_days = sorted_days;
    
end