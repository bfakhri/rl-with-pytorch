import gym
import math
import torch
import model
import numpy as np

# Using TensorBoardX for PyTorch: https://github.com/lanpa/tensorboard-pytorch
import tensorboardX as tbx


class ReplayBuffer():
    """This class stores gameplay data as torch tensors"""

    def __init__(self, buff_len, obs_shape, act_shape):
        # Number of steps in the buffer
        self.n = 0
        # Observations
        self.observations = torch.zeros(buff_len, *obs_shape) 
        # Action Probabilities
        self.action_probs = torch.zeros(buff_len, act_shape) 
        # Actions (a one-hot representation of what action was chosen)
        self.actions = torch.zeros(buff_len, act_shape) 
        # Rewards
        self.rewards = torch.zeros(buff_len)

    def append(self, ob, act_prob, act, r):
        """Adds new data to buffer"""

        self.observations[self.n] = ob
        self.action_probs[self.n] = act_prob
        self.actions[self.n] = act
        self.rewards[self.n] = r
        self.n += 1

    def discount(self, lambd):
        """Returns the discounted reward parameterized by 'lambd'"""

        # There is probably a more torch-like way to do this
        summer = 0
        for i in range(self.n):
            summer += math.pow(lambd, i)*self.rewards[i]

        return summer

# Instantiate the summary writer for tensorboard visualization
tb_writer = tbx.SummaryWriter()

# Instantiate the Environment
#env = gym.make('SpaceInvaders-v0')
env = gym.make('Pong-v0')

# Optimizer Params
LR = 0.01
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
nsteps_to_learn = 64

# Instantiate replay buffer
rp_buffer = ReplayBuffer(nsteps_to_learn, env.observation_space.shape, env.action_space.n) 

# Gather first observation
obs = env.reset()/255
# Training loop
while(episode < MAX_EPISODES):
    act_probs = model.act_probs(torch.from_numpy(obs))
    distrib = torch.distributions.Categorical(probs=act_probs)
    # Samples from the categorical distribution to determine action to take
    act_taken = distrib.sample()
    act_taken_v = torch.zeros(env.action_space.n)
    act_taken_v[act_taken] = 1

    # Takes the action
    observation, reward, done, info = env.step(act_taken.numpy()[0])

    # Perform book-keeping
    obs = observation/255   # Converts to float
    episode_reward += reward
    total_reward += reward
    episode_step += 1
    total_step += 1

    # Renders the game to screen
    #env.render()

    # Add experience to replay buffer
    rp_buffer.append(torch.from_numpy(obs), act_probs, act_taken_v, reward)
    
    # Learn from experience and clear rp buffer
    if(total_step%nsteps_to_learn == 0):
        # Calculates/Applies grads
        pl, cl, tl, dr = model.learn(rp_buffer)
        # Write outputs out for visualization
        tb_writer.add_scalar('Loss/PolicyLoss', pl, total_step) 
        tb_writer.add_scalar('Loss/CriticLoss', cl, total_step) 
        tb_writer.add_scalar('Loss/TotalLoss', tl, total_step) 
        tb_writer.add_scalar('Rewards/DiscountedReward', dr, total_step) 
        # Clears the replay buffer
        rp_buffer = ReplayBuffer(nsteps_to_learn, env.observation_space.shape, env.action_space.n) 

    # Episode has finished
    if(done):
        # Write out for tensorboard
        tb_writer.add_scalar('Rewards/EpisodeReward', episode_reward, total_step) 
        tb_writer.add_scalar('Rewards/RewardPerStep', episode_reward/episode_step, total_step) 
        tb_writer.add_scalar('Aux/EpisodeSteps', episode_step, total_step) 
        tb_writer.add_scalar('Aux/Progress', 1-episode/MAX_EPISODES, total_step) 
        print("Episode", str(episode), "/", str(MAX_EPISODES), "finished after", episode_step, "steps with", str(episode_reward), "reward")

        # Update/Reset metrics
        episode += 1
        episode_step = 0
        episode_reward = 0
        obs = env.reset()

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
