import numpy as np

# Preprocessing
STACK_SIZE = 4
FRAME_H = 84
FRAME_W = 84

# Model
STATE_SHAPE = [FRAME_H, FRAME_W, STACK_SIZE]
ACTION_SIZE = 4 # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
POSSIBLE_ACTIONS = [0, 1, 2, 3]

# Training
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 10_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.95

# Eps-greedy
EPS_START = 1
EPS_END = 1e-2
EPS_DECAY_RATE = 1e-4

# Memory for Experience Replay
MEMORY_SIZE = 50_000
PRETRAIN_LENGTH = BATCH_SIZE

# Fixed Q-targets
Q_TARGET_UPDATE_FREQ = 3_000

# Save the model every 100 episodes
SAVE_EVERY = 100
SAVE_PATH = './checkpoints'