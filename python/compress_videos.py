# Numerical tools
import numpy as np
import math

# Plotting tools
import matplotlib.pyplot as plt

# OS tools
import os
import tempfile
import time
import subprocess as sp
import getpass
import warnings
import argparse
import pickle

# Video tools
import moviepy
import moviepy.editor

# Remove warnings capture by moviepy (https://github.com/Zulko/moviepy/issues/1191)
import logging
logging.captureWarnings(False)

# Custom modules
import sys
sys.path.insert(0, '../python/')
import util
import session
import plot
import analysis

# Build combined nested dictionary via depth-first search.
def add_children(d, folder_id, names, children):
    # Dictionaries are not copied when indexed, so pass-by-reference will modify
    # original object.
    # Add ID and iterate through children. If leaf node, then will not enter for loop.
    d[names[folder_id]] = {'children': {}, 'id': folder_id}
    for child_id in children[folder_id]:
        add_children(d[names[folder_id]]['children'], child_id, names, children)

# Create helper method to convert string filepath to nested dictionary search.
def get_folder_from_tree(folder_tree, path):
    # Break path into list of subfolders.
    while path[0] == '/':
        path = path[1:]
    while path[-1] == '/':
        path = path[:-1]
    path = path.split('/')

    # Iterate through subfolders in path.
    folder = folder_tree
    for name in path[:-1]:
        folder = folder[name]['children']  
    return folder[path[-1]] # {'id': <id>, 'children': {...}}

# Main function to download, compress/reformat, and upload video.
warnings.filterwarnings('always', message='Reformatted video*', category=RuntimeWarning)
def process_video(filepath, 
                  file_id,
                  parent_id, *,
                  threads=None,
                  encoder='h264',
                  rtol=0.001,
                  verbose=True):
    # Initialize Google Drive interface.
    service = util.GoogleDriveService()
    
    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.mjpeg') as f_orig, \
         tempfile.NamedTemporaryFile(mode='w+b', suffix='.mp4') as f_comp:
        # Download video locally.
        service.download(file_id=file_id,
                         chunk_size=1024*1024*100,
                         file_object=f_orig)
              
        # Re-encode in h264 within mp4 container using ffmpeg.
        max_threads = len(os.sched_getaffinity(0))
        if threads is None:
            threads = max_threads
        cmd = ['ffmpeg',
               '-hide_banner',
               '-loglevel', 'warning',
               '-i', f_orig.name,
               '-codec', 'h264',
               '-copyts',
               '-vsync', 'vfr',
               '-threads', str(min(threads, max_threads)), # filter_threads for later versions
               f_comp.name]
        with sp.Popen(cmd) as p:
            p.wait()

        # Check that video metadata match.
        # HACK: cannot check duration of mjpeg via ffmpeg, so must put non-null value for t_stop.
        reader_orig = util.VideoReader(f_orig.name, t_stop='N/A')
        reader_comp = util.VideoReader(f_comp.name, t_stop=None)
        #assert (reader_orig.duration - reader_comp.duration)/reader_orig.duration <= rtol
        if reader_comp._duration < 1500.0: # warn if less than 25 minutes
            warnings.warn(f'Reformatted video at {filepath} unusually short '
                          + f'({reader_comp._duration} seconds).',
                          category=RuntimeWarning)
        assert (reader_orig._fps - reader_comp._fps)/reader_orig._fps <= rtol
        assert all([r1 == r2 for r1, r2 in zip(reader_orig._resolution, reader_comp._resolution)])

        # Upload compressed video to cloud.
        filename = filepath.split('/')[-1]
        content = service.upload(filepath=None,
                               file_stream=f_comp,
                               filename=filename.split('.')[0] + '.mp4',
                               folder_ids=parent_id,
                               mime_type='video/mp4',
                               chunk_size=1024*256*4,
                               verbose=verbose) 

        # Move uncompressed video to trash.
        result = service.trash_files(file_ids=file_id)

        # Sanity check.
        result = service.search_files(filename='.mp4',
                                      exact_match=False,
                                      mime_type='video/mp4',
                                      parent=parent_id,
                                      fields=['files(name,id)'])['files']
        assert (result['name'] == filename.split('.')[0] + '.mp4') and (result['id'] == content['id'])
                            
    return True

# Parse command-line arguments.
parser = argparse.ArgumentParser(description='Reformat behavior videos to mp4.')
parser.add_argument('-r', '--root-directory', default='My Drive/',
                    help='Topmost directory of Google Drive in which to search.')
parser.add_argument('-w', '--max-workers', type=int, default=2,
                    help='Maximum number of parallel processes to run at a given time.')
parser.add_argument('-e', '--encoder', default='h264',
                    help='Video encoder. Must be on list of available encoders for ffmpeg.')
parser.add_argument('-t', '--rtol', type=float, default=0.001,
                    help='Relative tolerance for comparing metadata of compressed video to'
                    + ' original video.')
parser.add_argument('-v', '--verbose', action='store_true', default=False,
                    help='Controls verbosity of output.')
args = parser.parse_args()

# # Initialize Google Drive interface.
# service = util.GoogleDriveService()

# # Start with root folder ('My Drive')
# root_id = service._service.files().get(fileId='root').execute()['id']
# folders = [{'name': 'My Drive',
#             'id': root_id}]

# # Grab rest of folder IDs for entire drive.
# max_page_limit = 100
# page = 0
# page_token = ''
# incomplete_search = False
# while (page_token is not None) and (page < max_page_limit):
#     # Get next page of folder IDs, names, and parents using the fields parameter.
#     # (Note this isn't on the main files.list() API.)
#     # https://developers.google.com/drive/api/guides/fields-parameter
#     result = service.search_files(mime_type='application/vnd.google-apps.folder',
#                                   pageSize=1000,
#                                   pageToken=page_token,
#                                   fields='nextPageToken, incompleteSearch, files(name,id,parents)')
#     folders += result['files']
    
#     # Get token for next page.
#     page_token = result.get('nextPageToken', None)
#     page += 1

# # Build two dictionaries:
# # 1. Maps parent_id --> child_ids {parent_id: [child_id, ...]}
# # 2. Maps folder_id --> name {folder_id: name}
# names = {folder['id']: folder['name'] for folder in folders}
# children = {folder['id']: [] for folder in folders}
# for folder in folders:
#     # Add folder to children of parent.
#     parents = folder.get('parents', [])
#     if len(parents) > 1:
#         raise SyntaxError('Multiple parents retrieved for folder {}.'.format(folder['name']))
#     elif len(parents) == 1:
#         children[parents[0]].append(folder['id'])

# # Build nested directionary of all folders in Google Drive.
# folder_tree = {}
# add_children(folder_tree, root_id, names, children)

# # Specify path of topmost directory.
# path = args.root_directory
# folder = get_folder_from_tree(folder_tree, path)

# # Find all mjpeg files to convert. Since we know the structure 
# # (freely-moving > animal > date > ___.mjpeg), we can simply iterate 
# # through the children of this directory.
# filepaths = []
# file_ids = []
# parent_ids = []
# for animal, subfolder in folder['children'].items():
#     for date, subsubfolder in subfolder['children'].items():
#         video_files = service.search_files(filename='.mjpeg',
#                                            exact_match=False,
#                                            mime_type='image/jpeg',
#                                            parent={'id': subsubfolder['id']},
#                                            fields=['files(name,id)'])['files']
#         for video_file in video_files:
#             filepaths.append(f'{animal}/{date}/{video_file["name"]}')
#             file_ids.append(video_file['id'])
#             parent_ids.append(subsubfolder['id'])

# # Cache results for debugging.
# for name, obj in zip(['filepaths', 'file_ids', 'parent_ids'],
#                       [filepaths, file_ids, parent_ids]):
#     with open(f'/home/james/Desktop/debug/{name}.p', 'wb') as f:
#         pickle.dump(obj, f)

# Open cached results.
with open('/home/james/Desktop/debug/filepaths.p', 'rb') as f:
    filepaths = pickle.load(f)
with open('/home/james/Desktop/debug/file_ids.p', 'rb') as f:
    file_ids = pickle.load(f)
with open('/home/james/Desktop/debug/parent_ids.p', 'rb') as f:
    parent_ids = pickle.load(f)

# Settings
max_workers = args.max_workers # consider available storage as mpjpeg files can exceed 20 GB
threads_per_worker = int((os.cpu_count()-2)/max_workers) # os.cpu_count() gives number of logical CPUs (aka threads)
encoder = args.encoder
rtol = args.rtol
verbose = args.verbose

# Run video processes in parallel.
# BUG: Get attribute error on Mac. Need to test on Linux.
mpm = util.MultiprocessManager(max_workers=max_workers, verbose=True)
mpm.run(process_video, 
        filepaths,
        file_ids,
        parent_ids,
        threads=threads_per_worker,
        encoder=encoder,
        rtol=rtol,
        verbose=verbose,
        names=filepaths)