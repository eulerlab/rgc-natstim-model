import os
import numpy as np
from .paths import base_directory
DUR_MOVIE = 18450
RANDOM_SEQUENCES_PATH = os.path.join(base_directory, 'data', 'movie', 'RandomSequences.npy')
GROUP_INFO_PATH = os.path.join(base_directory, 'data', 'movie', 'group_info.pkl')
#DATA_PATH = r'/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/context_change_detection/'
START_INDICES = np.arange(150, 18450, 150)
NUM_CLIPS_TOTAL = 123
NUM_CLIPS = 108
NUM_VAL_CLIPS = 15
CLIP_DUR = 150
MOVIE_DUR = int(NUM_CLIPS_TOTAL*CLIP_DUR)