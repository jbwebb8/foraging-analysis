import argparse
import json
import sys, os
from util import find_files, find_data
from session import FreeSession
from plot import *

### Initial setup ###
# Command line arguments
parser = argparse.ArgumentParser(description='Analysis for patch-based foraging task.')
parser.add_argument('config_file', help='config file for analysis settings')
parser.add_argument('--check-filelist', action='store_true', default=False,
                    help='check filenames to be analyzed')
args = parser.parse_args()

# Grab arguments
config_file = args.config_file
check_filelist = args.check_filelist

# Experiment settings
if not config_file.lower().endswith('.json'): 
    raise Exception('%s is not a valid JSON file.' % config_file)
config = json.loads(open(config_file).read())
mouse_ids  = config['global']['mouse_ids']
exp_name = config['global']['exp_name']
results_dir = config['global']['results_directory']
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

# Create Session objects from filenames
root_dir = config['global']['data_directory'] # root directory of all data files
exclude_strs = config['global']['exclude_strings']
sessions = {}
days = {}
files = find_files(root_dir, [])
for mouse_id in mouse_ids:
    # Get filenames and training days
    filelist, training_days = find_data(mouse_id, files, exclude_strs=exclude_strs)

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
print('Checking for data in %s...' % results_dir)
for mouse_id in mouse_ids:
    print('Loading mouse %s:' % mouse_id)
    for sess, day in zip(sessions[mouse_id], days[mouse_id]):
        print('Loading session %02d...' % day, end=' ')
        try:
            filepath = results_dir + mouse_id + '\\%02d.p' % day 
            sess.load(filepath)
            print('loaded.')
        except FileNotFoundError:
            print('file not found.') 
    print()

# Define periodic save function
def save_sessions(mouse_id):
    print('Saving sessions for mouse %s...' % mouse_id)
    mouse_dir = results_dir + mouse_id + '\\'
    if not os.path.isdir(mouse_dir):
        os.mkdir(mouse_dir)
    for sess, day in zip(sessions[mouse_id], days[mouse_id]):
        sess.save(mouse_dir + '%02d.p' % day)

# Create plot object
plot_settings = config['plot_settings']
plot = Plotter(results_dir, **plot_settings)

### Harvest rate analysis ###
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
        print('Processing session %d...' % day)

        try:
            hr_obs[mouse_id].append(sess.get_harvest_rate(metric='observed', per_patch=True))
            hr_max[mouse_id].append(sess.get_harvest_rate(metric='max', per_patch=True))
            hr_opt[mouse_id].append(sess.get_harvest_rate(metric='optimal', per_patch=True))
        except Warning:
            hr_obs[mouse_id].append(np.array([np.nan]))
            hr_max[mouse_id].append(np.array([np.nan]))
            hr_opt[mouse_id].append(np.array([np.nan]))
        
        # Clear data for memory management
        sess.clear_data()

    # Save updated session variables
    #save_sessions(mouse_id)
    print()     

plot.plot_harvest_rates(hr_obs, days, center='mean', plot_points=True)

# Save figure
plot.save_figure('hr_vs_day.pdf')
print('Harvest rate analysis done.')