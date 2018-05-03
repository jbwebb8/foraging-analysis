% Test block
filelist = importdata('G:\My Drive\Projects\foraging\data\matlist.txt');
t_p = zeros(size(filelist, 1), 2);
t_t = zeros(size(filelist, 1), 2);
r_p = zeros(size(filelist, 1), 2);

for i = 1:size(filelist)
    % Get filename
    filename = filelist{i};
    fprintf('Processing file %s\n', filename);
    
    % Get patch statistics
    fprintf('Getting patch data...\n');
    [t_p_i, t_t_i, r_p_i] = get_patch_data(filename, false);
    t_p(i, :) = [mean(t_p_i), std(t_p_i)];
    t_t(i, :) = [mean(t_t_i), std(t_t_i)];
    r_p(i, :) = [mean(r_p_i), std(r_p_i)];
    
    % Save data
    [start_idx, end_idx] = regexp(filename, 'j[0-9]+[a-z][0-9]+_d[0-9]+_');
    new_filename = [filename(start_idx:end_idx), 'patch_data.mat'];
    fprintf('Saving to %s\n', new_filename);
    save(new_filename, 't_p', 't_t', 'r_p');
end