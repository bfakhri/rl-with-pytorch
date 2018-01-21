import gym
import model
import numpy as np

#class ReplayBuffer():
    def __init__(self, obs_shape):
        # Number of steps in the buffer
        self.n = 0
        # Observations
        shape_with_time = (1,) + obs_shape
        self.obs = np.empty(shape_with_time, dtype=np.float)
        # Rewards
        self.r = np.empty((1,), dtype=np.float)




# Instantiate the Environment
env = gym.make('SpaceInvaders-v0')

# Instantiate the model
model = model.Model(env.observation_space.shape, env.action_space.n) 

# Gather first observation
obs = env.reset()

# Training Limits
MAX_EPISODES = 10

# Training Loop
episode = 0
episode_step = 0
total_step = 0
episode_reward = 0
total_reward = 0
while(episode < MAX_EPISODES):
    action = model.act(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    total_reward += reward
    episode_step += 1
    total_step += 1
    env.render()

    # Episode has finished
    if(done):
        print("Episode", str(episode), "/", str(MAX_EPISODES), "finished after", episode_step, "steps with", str(episode_reward), "reward")
        episode += 1
        episode_step = 0
        episode_reward = 0
        obs = env.reset()

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
