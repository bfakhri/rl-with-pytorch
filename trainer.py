import gym
import math
import torch
import model
import numpy as np

# Holds data as torch tensors 
class ReplayBuffer():
    def __init__(self, buff_len, obs_shape, act_shape):
        # Number of steps in the buffer
        self.n = 0
        # Observations
        self.observations = torch.zeros(buff_len, *obs_shape) 
        # Actions
        self.actions = torch.zeros(buff_len, act_shape) 
        # Rewards
        self.rewards = torch.zeros(buff_len)

    def append(self, ob, act, r):
        self.observations[self.n] = ob
        self.actions[self.n] = act
        self.rewards[self.n] = r
        self.n += 1

    def discount(self, lambd):
        summer = 0
        for i in range(self.n):
            summer += math.pow(lambd, i)*self.rewards[i]

        return summer

# Instantiate the Environment
env = gym.make('SpaceInvaders-v0')


# Optimizer Params
LR = 0.0001
MOMENTUM = 0.5

# Instantiate the model and optimizer
model = model.Model(env.observation_space.shape, env.action_space.n, LR, MOMENTUM) 


# Training Limits
MAX_EPISODES = 100000

# Training Loop
episode = 0
episode_step = 0
total_step = 0
episode_reward = 0
total_reward = 0
nsteps_to_learn = 10 

# Instantiate replay buffer
rp_buffer = ReplayBuffer(nsteps_to_learn, env.observation_space.shape, env.action_space.n) 

# Gather first observation
obs = env.reset()/255
# Training loop
while(episode < MAX_EPISODES):
    act_probs = model.act_probs(torch.from_numpy(obs))
    val, idx = torch.max(act_probs,dim=1)
    observation, reward, done, info = env.step(idx.numpy()[0])
    print(idx.numpy()[0])
    obs = observation/255   # Converts to float
    episode_reward += reward
    total_reward += reward
    episode_step += 1
    total_step += 1
    #env.render()
    # Add experience to replay buffer
    rp_buffer.append(torch.from_numpy(obs), act_probs, reward)
    
    # Learn from experience and clear rp buffer
    if(total_step%nsteps_to_learn == 0):
        # Calculates/Applies grads
        model.learn(rp_buffer)
        # Clears the replay buffer
        rp_buffer = ReplayBuffer(nsteps_to_learn, env.observation_space.shape, env.action_space.n) 

    # Episode has finished
    if(done):
        print("Episode", str(episode), "/", str(MAX_EPISODES), "finished after", episode_step, "steps with", str(episode_reward), "reward")
        episode += 1
        episode_step = 0
        episode_reward = 0
        obs = env.reset()

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
