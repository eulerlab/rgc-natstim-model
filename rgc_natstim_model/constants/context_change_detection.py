import numpy as np

DUR_MOVIE = 18450
RANDOM_SEQUENCES_PATH = r'/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/data/RandomSequences.npy'
GROUP_INFO_PATH = r'/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/data/group_info.pkl'
DATA_PATH = r'/gpfs01/euler/data/SharedFiles/projects/Hoefling2024/context_change_detection/'
START_INDICES = np.arange(150, 18450, 150)
NUM_CLIPS_TOTAL = 123
NUM_CLIPS = 108
NUM_VAL_CLIPS = 15
CLIP_DUR = 150
MOVIE_DUR = int(NUM_CLIPS_TOTAL*CLIP_DUR)