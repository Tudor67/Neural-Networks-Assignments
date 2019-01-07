import numpy as np

# Preprocessing
STACK_SIZE = 4 

# Model
STATE_SHAPE = [110, 84, STACK_SIZE]
ACTION_SIZE = 4 # ['BUTTON', '', 'LEFT', 'RIGHT']
# Hot encoded version of our actions
GENERAL_ACTIONS = np.identity(8, dtype=int)
POSSIBLE_ACTIONS = GENERAL_ACTIONS[[0,1,6,7]]

# Training
NUM_EPISODES = 300
MAX_STEPS_PER_EPISODE = 50_000
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.9

# Eps-greedy
EPS_START = 1
EPS_END = 1e-2
EPS_DECAY_RATE = 2e-5

# Memory for Experience Replay
MEMORY_SIZE = 50_000
PRETRAIN_LENGTH = BATCH_SIZE

# Fixed Q-targets
Q_TARGET_UPDATE_FREQ = 10_000

# Save the model every 50 episodes
SAVE_EVERY = 50
SAVE_PATH = './checkpoints'