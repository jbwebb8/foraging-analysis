import argparse
import json
import sys, os
from util import find_files, find_data, recursive_dict_search
from analysis import get_lick_stats
from session import FreeSession
from plot import *

########################### Initial setup #####################################
# Command line arguments
parser = argparse.ArgumentParser(description='Analysis for patch-based foraging task.')
parser.add_argument('config_file', help='config file for analysis settings')
parser.add_argument('--check-filelist', action='store_true', default=False,
                    help='check filenames to be analyzed')
args = parser.parse_args()

# Grab arguments
config_file = args.config_file
check_filelist = args.check_filelist

# Load and configure experiment settings file
if not config_file.lower().endswith('.json'): 
    raise Exception('%s is not a valid JSON file.' % config_file)
config = json.loads(open(config_file).read())
recursive_dict_search(config, "None", None)
recursive_dict_search(config, "True", True)
recursive_dict_search(config, "False", False)

# Set experiment settings
mouse_ids  = config['global']['mouse_ids']
exp_name = config['global']['exp_name']
sess_dir = config['global']['session_directory']
results_dir = config['global']['results_directory']
save_updates = config['global']['save_sessions']

# Create directories if needed
if not os.path.isdir(sess_dir):
    os.mkdir(sess_dir)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
if not os.path.isdir(results_dir + exp_name + '/'):
    os.mkdir(results_dir + exp_name + '/') # population results
for mouse_id in mouse_ids:
    if not os.path.isdir(results_dir + mouse_id + '/'):
        os.mkdir(results_dir + mouse_id + '/') # individual results

# Create Session objects from filenames
root_dir = config['global']['data_directory'] # root directory of all data files
exclude_strs = config['global']['exclude_strings']
sessions = {}
days = {}
files = find_files(root_dir, [])
for mouse_id in mouse_ids:
    # Get filenames and training days
    filelist, training_days = find_data(mouse_id, files, exclude_strs=exclude_strs)

    # Filter by global training range
    if len(config['global'].get('day_range', [])) > 0:
        day_range = config['global']['day_range']
        keep_idx = np.logical_and(training_days >= day_range[0], 
                                  training_days <= day_range[1])
        filelist = [f for f, keep in zip(filelist, keep_idx) if keep]
        training_days = training_days[keep_idx]

    # Check filelist
    print('Files to analyze (mouse %s):' % mouse_id)
    print('Day  Filename')
    for day, f in zip(training_days, filelist):
        print('%2d: ' % day, f)
    print()
    
    # Store as sessions, days
    sessions[mouse_id] = [FreeSession(filename) for filename in filelist]
    days[mouse_id] = training_days

# Check filenames if flagged
if check_filelist:
    print('Is the above filelist correct? [y/n]')
    while True:
        ans = input()
        if ans.lower() in ['y', 'yes']:
            break
        elif ans.lower() in ['n', 'no']:
            sys.exit()
        else:
            print('Invalid response. Please type \'y\' or \'n\'.')

# Load session data
print('Checking for data in %s...' % sess_dir)
for mouse_id in mouse_ids:
    print('Loading mouse %s:' % mouse_id)
    for sess, day in zip(sessions[mouse_id], days[mouse_id]):
        print('Loading session %02d...' % day, end=' ')
        try:
            filepath = sess_dir + mouse_id + '/%02d.p' % day 
            sess.load(filepath, ignore_keys=['data_names', 'var_names'])
            print('loaded.')
        except FileNotFoundError:
            print('file not found.') 
    print()

# Define periodic save function
def save_sessions(mouse_id):
    print('Saving sessions for mouse %s...' % mouse_id)
    mouse_dir = sess_dir + mouse_id + '/'
    if not os.path.isdir(mouse_dir):
        os.mkdir(mouse_dir)
    for sess, day in zip(sessions[mouse_id], days[mouse_id]):
        sess.save(mouse_dir + '%02d.p' % day)

# Create plot object
plot_settings = config['plot_settings']
plot = Plotter(**plot_settings)

# Save config file for reference
with open(results_dir + exp_name + '/config.json', 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

####################### Harvest rate analysis #################################
if config['harvest_rate']['analyze']:
    print('HARVEST RATE ANALYSIS')

    # Grab harvest rate over sessions
    hr_obs = {} # observed rate
    hr_max = {} # maximum rate given leaving decisions
    hr_opt = {} # optimal rate (MVT) given environment
    for mouse_id in mouse_ids:
        print('Analyzing mouse %s:' % mouse_id)

        # Placeholders
        hr_obs[mouse_id] = []
        hr_max[mouse_id] = []
        hr_opt[mouse_id] = []

        for sess, day in zip(sessions[mouse_id], days[mouse_id]):
            print('Processing session %d...' % day, end=' ')

            try:
                hr_obs[mouse_id].append(sess.get_harvest_rate(metric='observed', per_patch=True))
                hr_max[mouse_id].append(sess.get_harvest_rate(metric='max', per_patch=True))
                hr_opt[mouse_id].append(sess.get_harvest_rate(metric='optimal', per_patch=True))
            except UserWarning as w: # unanalyzable session (e.g. not enough patches)
                print(w)
                hr_obs[mouse_id].append(np.array([np.nan]))
                hr_max[mouse_id].append(np.array([np.nan]))
                hr_opt[mouse_id].append(np.array([np.nan]))
            
            # Clear data for memory management
            sess.clear_data()
            print('done.')

        # Save updated session variables
        if save_updates:
            save_sessions(mouse_id)
        print()     

    # Plot individual learning curves
    for mouse_id in mouse_ids:
        print('Plotting harvest rate for mouse %s...' % mouse_id, end=' ')
        plot.plot_harvest_rates(days[mouse_id], hr_obs[mouse_id], 
                                **config['harvest_rate']['kwargs'])
        plot.save_figure(results_dir + mouse_id + '/hr_vs_day.pdf')
        plot.plot_harvest_diffs(days[mouse_id], hr_obs[mouse_id], hr_opt[mouse_id], 
                                hr_max[mouse_id], **config['harvest_rate']['kwargs'])
        plt.savefig(results_dir + mouse_id + '/hr_diff_vs_day.pdf')
        print('done.')

    # Plot population learning curve
    print('Plotting population learning curve...', end=' ')
    plot.plot_harvest_rates(days, hr_obs, **config['harvest_rate']['kwargs'])
    plot.save_figure(results_dir + exp_name + '/hr_vs_day.pdf')
    plot.plot_harvest_diffs(days, hr_obs, hr_opt, hr_max, **config['harvest_rate']['kwargs'])
    plt.savefig(results_dir + exp_name + '/hr_diff_vs_day.pdf')
    print('done.')


############################ MVT analysis #####################################
if config['mvt']['analyze']:
    print('MVT ANALYSIS')

    # Grab patch residence and travel times
    t_p_obs = {}
    t_p_opt = {}
    t_t_obs = {}
    t_t_opt = {}
    for mouse_id in mouse_ids:
        print('Analyzing mouse %s:' % mouse_id)
        
        # Placeholders
        t_p_obs[mouse_id] = []
        t_p_opt[mouse_id] = []
        t_t_obs[mouse_id] = []
        t_t_opt[mouse_id] = []
        
        for sess, day in zip(sessions[mouse_id], days[mouse_id]):
            print('Processing session %d... ' % day, end=' ')
            
            try:
                t_p_obs[mouse_id].append(sess.get_patch_durations())
                _, _, t_p_opt_, t_t_opt_ = sess.get_harvest_rate(metric='optimal', return_all=True)
                t_p_opt[mouse_id].append(np.asarray([t_p_opt_]))
                t_t_obs[mouse_id].append(sess.get_interpatch_durations())
                t_t_opt[mouse_id].append(np.asarray([t_t_opt_]))
            except UserWarning as w: # unanalyzable session
                print(w)
                t_p_obs[mouse_id].append(np.asarray([np.nan]))
                t_p_opt[mouse_id].append(np.asarray([np.nan]))
                t_t_obs[mouse_id].append(np.asarray([np.nan]))
                t_t_opt[mouse_id].append(np.asarray([np.nan]))

            # Clear data for memory management
            sess.clear_data()
            print('done.')
        
        # Save updated session variables
        if save_updates:
            save_sessions(mouse_id)
        print()

    # Plot individual learning curves
    for mouse_id in mouse_ids:
        print('Plotting residence/travel times for mouse %s...' % mouse_id, end=' ')
        plot.plot_residence_times(days[mouse_id], t_p_obs[mouse_id], 
                                t_p_opt[mouse_id], **config['mvt']['kwargs'])
        plot.save_figure(results_dir + mouse_id + '/t_p_vs_day.pdf')
        plot.plot_travel_times(days[mouse_id], t_t_obs[mouse_id], **config['mvt']['kwargs'])
        plt.savefig(results_dir + mouse_id + '/t_t_vs_day.pdf')
        print('done.')

    # Plot population learning curve
    print('Plotting population learning curve...', end=' ')
    plot.plot_residence_times(days, t_p_obs, t_p_opt, **config['mvt']['kwargs'])
    plot.save_figure(results_dir + exp_name + '/t_p_vs_day.pdf')
    plot.plot_travel_times(days, t_t_obs, **config['mvt']['kwargs'])
    plt.savefig(results_dir + exp_name + '/t_t_vs_day.pdf')
    print('done.')


############################ Lick analysis ####################################
if config['lick']['analyze']:
    print('LICK ANALYSIS')

    # Placeholders
    lick_stats = {'n_total': {},
                'n_patch': {},
                'n_interpatch': {},
                'f_patch': {},
                'f_interpatch': {}}

    # Get stats over sessions
    for mouse_id in mouse_ids:
        print('Analyzing mouse %s:' % mouse_id)
        
        # Placeholders
        for k in lick_stats.keys():
            lick_stats[k][mouse_id] = []
        
        for sess, day in zip(sessions[mouse_id], days[mouse_id]):
            print('Processing session %d...' % day, end=' ')
            
            # Add stats in format: lick_stats[key][mouse]
            try:
                lick_stats_ = get_lick_stats(sess)
                for k in lick_stats.keys():
                    lick_stats[k][mouse_id].append(lick_stats_[k])
            except UserWarning as w: # unanalyzable session
                print(w)
                for k in lick_stats.keys():
                    lick_stats[k][mouse_id].append(np.asarray([np.nan]))
            
            # Clear data for memory management
            sess.clear_data()
            print('done.')
        
        # Save updated session variables
        if save_updates:
            save_sessions(mouse_id)
        print()
            
    print('Done.')