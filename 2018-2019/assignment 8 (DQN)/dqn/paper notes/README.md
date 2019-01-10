# Paper notes

## Some notes from the [DeepMind paper](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
* RL is known to be unstable when a nonlinear function approximator such as NN is used to represent the action-value/Q function. This instability has several causes:
  - the correlations present in the sequence of observations (small updates to Q may significantly change the policy and therefore change the data distribution);
  - the correlations between the action-values/Q and the target values ![q-target](https://latex.codecogs.com/gif.latex?%24r%20&plus;%20%5Cgamma%20%5Cmax%20%5Climits_%7Ba%27%7D%20Q%28s%27%2C%20a%27%29%24).
  
* These instabilities are addressed with a novel variant of Q-learning which uses 2 key ideas:
  - `Experience replay` that randomizes over the data, thereby removing correlations in the observation sequence and smoothing over the changes in the data distribution;
  - `Fixed Q-targets` that are only periodically updated, thereby reducing correlations with the target.

* Q-learning:
  - __model-free__: it solves the RL task directly using samples from the emulator, without explicitly estimating the reward and transition dynamics P(r, s' | s, a) (it does not require a model of the environment);
  - __off-policy__: it learns about the greedy policy ![greedy-policy](https://latex.codecogs.com/gif.latex?%24a%20%3D%20argmax_%7Ba%27%7D%20Q%28s%2C%20a%27%3B%20%5Ctheta%29%24), while following a behaviour distribution that ensures adequate exploration of the state space: eps-greedy policy. The main idea is that a different policy is used for value evaluation than what is used to select the next action.


> __On-policy__ methods estimate the value of a policy while using it for control.

> In __off-policy__ methods, the policy used to generate behaviour, called the _behaviour policy_, may be unrelated to the policy that is evaluated and improved, called the _estimation policy_.

> An advantage of this separation is that the _estimation policy_ may be deterministic (e.g. greedy), while the _behaviour policy_ can continue to sample all possible actions (eps-greedy).

## References
1. [Human-level control through deep reinforcement learning (DeepMind paper)](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
2. [Off-policy vs On-policy learning](https://stats.stackexchange.com/questions/184657/what-is-the-difference-between-off-policy-and-on-policy-learning)

