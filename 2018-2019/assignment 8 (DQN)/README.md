# Assignment 8 (2018-2019)
__Deep Q-Networks (DQNs)__  
I trained DQNs on Breakout and SpaceInvaders-Atari2600 (details and results for SpaceInvaders can be found in `./tutorials`).

## Some notes from the [DeepMind paper](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
* RL is known to be unstable when a nonlinear function approximator such as NN is used to represent the action-value/Q function. This instability has several causes:
  - the correlations present in the sequence of observations (small updates to Q may significantly change the policy and therefore change the data distribution);
  - the correlations between the action-values/Q and the target values $r + \gamma \max \limits_{a'} Q(s', a')$.
  
* These instabilities are addressed with a novel variant of Q-learning which uses 2 key ideas:
  - `Experience replay` that randomizes over the data, thereby removing correlations in the observation sequence and smoothing over the changes in the data distribution;
  - `Fixed Q-targets` that are only periodically updated, thereby reducing correlations with the target.

* Q-learning:
  - __model-free__: it solves the RL task directly using samples from the emulator, without explicitly estimating the reward and transition dynamics $P(r, s'| s, a)$ (it does not require a model of the environment);
  - __off-policy__: it learns about the greedy policy $a = argmax_{a'}  Q(s, a'; \theta)$, while following a behaviour distribution that ensures adequate exploration of the state space: $\epsilon$-greedy policy. The main idea is that a different policy is used for value evaluation than what is used to select the next action.


> __On-policy__ methods estimate the value of a policy while using it for control.
> In __off-policy__ methods, the policy used to generate behaviour, called the _behaviour policy_, may be unrelated to the policy that is evaluated and improved, called the _estimation policy_.
> An advantage of this seperation is that the _estimation policy_ may be deterministic (e.g. greedy), while the _behaviour policy_ can continue to sample all possible actions ($\epsilon$-greedy).

## CNN architecture and training details

## Results

## References
1. [OpenAI Gym (Docs)](https://gym.openai.com/docs/)
2. [OpenAI Gym (Value iteration and Q-learning)](https://www.kaggle.com/charel/learn-by-example-reinforcement-learning-with-gym)
3. [Intro to RL - Part 1 (What is RL ?)](https://medium.com/@adeshg7/introduction-to-reinforcement-learning-part-1-dbfd19c28a30)
4. [Intro to RL - Part 2 (MDPs and Q-learning)](https://medium.com/@adeshg7/introduction-to-reinforcement-learning-part-2-74e0a3fad9d3)
5. [Intro to RL - Part 3 (Coding Q-learning)](https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0)
6. [An introduction to RL - P1 (MC vs TD, 3 approaches to RL: value-based, policy-based, model-based)](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)
7. [An introduction to RL - P2 (Diving deeper into RL with Q-Learning)](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe)
8. [An introduction to RL - P3 (Deep Q-Learning)](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)
9. [Welcome to Deep RL: Learning part](https://towardsdatascience.com/welcome-to-deep-reinforcement-learning-part-1-dqn-c3cab4d41b6b)
10. [RL lecture (structure of RL)](http://bicmr.pku.edu.cn/~wenzw/bigdata/MDP.pdf)
11. [RL (general ideas)](https://towardsdatascience.com/my-journey-to-reinforcement-learning-part-0-introduction-1e3aec1ee5bf)
12. [Human-level control through deep reinforcement learning (DeepMind paper)](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
13. [Off-policy vs On-policy learning](https://stats.stackexchange.com/questions/184657/what-is-the-difference-between-off-policy-and-on-policy-learning)

