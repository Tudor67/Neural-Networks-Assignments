# Game
GAME_NAME = 'Pong-v0'

# Preprocessing
STACK_SIZE = 4
FRAME_H = 84
FRAME_W = 84

# Model
STATE_SHAPE = [FRAME_H, FRAME_W, STACK_SIZE]

# Training
NUM_EPISODES = 1_000
BATCH_SIZE = 32
# Discount factor
GAMMA = 0.99 
# RMSProp
LEARNING_RATE = 2.5e-4
DECAY = 0.95
MOMENTUM = 0
EPSILON = 1e-5
CENTERED = True

# Eps-greedy (linear decaying)
EPS_START = 1
EPS_END = 1e-2
DECAY_STEPS = 300_000
DECAY_STEP_LENGTH = (EPS_START - EPS_END) / DECAY_STEPS

# Memory for Experience Replay
MEMORY_SIZE = 300_000
PRETRAIN_LENGTH = 30_000

# Fixed Q-targets
Q_TARGET_UPDATE_FREQ = 3_000

# Save the model every 50 episodes
SAVE_EVERY = 50
SAVE_PATH = './checkpoints'