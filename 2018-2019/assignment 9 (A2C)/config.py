# Game
GAME_NAME = 'Pong-v0'

# Preprocessing
STACK_SIZE = 4
FRAME_H = 84
FRAME_W = 84

# Model
STATE_SHAPE = [FRAME_H, FRAME_W, STACK_SIZE]

# Training
NUM_EPISODES = 100
# Discount factor
GAMMA = 0.99 
# RMSProp
LEARNING_RATE = 2.5e-4
DECAY = 0.95
MOMENTUM = 0
EPSILON = 1e-5
CENTERED = True

# Save the model every 50 episodes
SAVE_EVERY = 50
SAVE_PATH = './checkpoints'