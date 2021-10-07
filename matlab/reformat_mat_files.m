listpath = '/home/james/Desktop/filelist.txt';
localdir = '/media/james/data/foraging/head_fixed';
filelist = importdata(listpath);
for i=1:numel(filelist)
    % Load data from filepath
    filepath = filelist{i};
    vars = who('-file', filepath); % inspect for variables to save
    load(filepath);
    
    % Save to new file with version 7.3 formatting
    [pathstr, name, ext] = fileparts(filepath);
    mouse_id = strsplit(name, '_');
    mouse_id = mouse_id{1};
    
    newFilepath = fullfile(localdir, mouse_id, strcat(name, ext));
    save(newFilepath, vars{:}, '-v7.3');
    
    % Clear loaded variables
    clearvars(vars{:});
end