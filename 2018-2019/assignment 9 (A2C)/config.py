# Game
GAME_NAME = 'Pong-v0'

# Preprocessing
STACK_SIZE = 4
FRAME_H = 84
FRAME_W = 84

# Model
STATE_SHAPE = [FRAME_H, FRAME_W, STACK_SIZE]

# Training
NUM_EPISODES = 2500
# Discount factor
GAMMA = 0.99 
# RMSProp
LEARNING_RATE = 2.5e-4

# Save the model every 50 episodes
SAVE_EVERY = 50
SAVE_PATH = './checkpoints'