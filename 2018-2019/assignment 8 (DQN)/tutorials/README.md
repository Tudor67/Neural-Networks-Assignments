# Tutorial 1: Value iteration
`OpenAI_Gym_1_Value_Iteration.ipynb`

# Tutorial 2: Q-learning
`OpenAI_Gym_2_Q_learning.ipynb`

# Tutorial 3: DQN on SpaceInvaders-Atari2600
0. Roms:  
    a. Download zip file from http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html  
    b. Open Roms.rar > ROMS.rar and find Space Invaders (1980) XXXXXX  
    c. Extract all matches (there are 5 of them) into your destination folder  
    d. python -m retro.import . (don't forget the point)  
1. `OpenAI_Gym_3_DQN_train.ipynb`
2. `OpenAI_Gym_3_DQN_test.ipynb`
3. `Save_train_animations.ipynb`

## CNN architecture and training details
![dqn_details](./images/dqn_details.png)

## DQN experiments on SpaceInvaders-Atari2600
![rmsprop_plots](./results/rmsprop_plots.png)

| Optimizer | RMSProp(lr=2.5e-4) | RMSProp(lr=1e-3) | RMSProp(lr=1e-2) |
| :---: | :---: | :---: | :---: |
| Train | ![rmsprop_lr_25e-5_train_episode_3](./results/rmsprop_lr_25e-5_train_episode_3_reward_805.gif) | ![rmsprop_lr_1e-3_train_episode_16](./results/rmsprop_lr_1e-3_train_episode_16_reward_770.gif) | ![rmsprop_lr_1e-2_train_episode_11](./results/rmsprop_lr_1e-2_train_episode_11_reward_835.gif)
| Test/Eval | ![rmsprop_lr_25e-5_test_episode_100](./results/rmsprop_lr_25e-5_test_episode_100_reward_170.gif) | ![rmsprop_lr_1e-3_test_episode_100](./results/rmsprop_lr_1e-3_test_episode_100_reward_325.gif) | ![rmsprop_lr_1e-2_test_episode_100](./results/rmsprop_lr_1e-2_test_episode_100_reward_0.gif)

