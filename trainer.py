import gym
import math
import torch
import model
import numpy as np
from torch.autograd import Variable
# Using TensorBoardX for PyTorch: https://github.com/lanpa/tensorboard-pytorch
import tensorboardX as tbx

# Instantiate the summary writer for tensorboard visualization
#tb_writer = tbx.SummaryWriter(comment="RunDescription")
tb_writer = tbx.SummaryWriter(comment="DetachedAdv")

cuda = True
#cuda = False 

def torcher(arr_to_torch):
    """Converts the input to the proper torch tensor, whether that is 
    a cuda torch tensor or a regular torch tensor"""
    if(cuda):
        return torch.from_numpy(arr_to_torch).cuda().float()
    else:
        return torch.from_numpy(arr_to_torch).float()


class ReplayBuffer():
    """This class stores gameplay data as torch tensors"""

    def __init__(self, buff_len, obs_shape, act_shape):
        # Length of buffer
        self.buff_len = buff_len
        # Number of steps in the buffer
        self.n = 0
        # Observations
        self.observations = torcher(np.zeros((buff_len,)+obs_shape))
        # Action Probabilities
        self.action_probs = torcher(np.zeros((buff_len,)+(act_shape,)))
        # Actions (a one-hot representation of what action was chosen)
        self.actions = torcher(np.zeros((buff_len,)+(act_shape,)))
        # Rewards
        self.rewards = torcher(np.zeros(buff_len))
        # Dones 
        self.dones = torcher(np.zeros(buff_len))

    def append(self, ob, act_prob, act, r, done):
        """Adds new data to buffer"""

        self.observations[self.n] = ob
        self.action_probs[self.n] = act_prob
        self.actions[self.n] = act
        self.rewards[self.n] = r
        self.dones[self.n] = done 
        self.n += 1

    def discount(self, lambd):
        """Returns the discounted reward parameterized by 'lambd'"""

        # There is probably a more torch-like way to do this
        discounted_rewards = torcher(np.zeros(self.buff_len))
        summer = 0
        for i in range(self.n):
            summer += math.pow(lambd, i)*self.rewards[i]
            discounted_rewards[i] = summer
            if(self.dones[i] == 1):
                summer = 0
        return discounted_rewards

    def actions_scalar(self):
        """Returns array of scalars corresponding to the actions taken"""

        values, actions_s = torch.max(self.actions, dim=1)
        return actions_s

# Instantiate the Environment
#env_str = 'SpaceInvaders-v0'
env_str = 'Pong-v0'
env = gym.make(env_str)

# Optimizer Params
LR = 0.00001
MOMENTUM = 0.5
tb_writer.add_scalar('HyperParams/LR', LR, 0) 
tb_writer.add_scalar('HyperParams/Momentum', MOMENTUM, 0) 

# Instantiate the model and optimizer
model = model.Model(env.observation_space.shape, env.action_space.n, LR, MOMENTUM, cuda) 

# Training Limits
MAX_EPISODES = 100000

# Training Loop
episode = 0             # Current episode number
episode_step = 0        # Total steps during current episode
total_step = 0          # Total steps trained on
episode_reward = 0      # Total reward during episode
total_reward = 0        # Running count of the reward obtained
nsteps_to_learn = 64    # Number of steps to perform backprop on
validate_freq = 20      # Number of episodes between validation phase

# Instantiate replay buffer
rp_buffer = ReplayBuffer(nsteps_to_learn, env.observation_space.shape, env.action_space.n) 

# Gather first observation
np_obs = env.reset()/255
obs = torcher(np_obs)

# Add the computational graph to TensorBoard
tb_writer.add_graph(model, (Variable(obs.unsqueeze(0)), ))

# Training loop
while(episode < MAX_EPISODES):
    # Asks the model for the action
    act_taken, act_taken_v, act_probs = model.act_stochastic(obs)

    # Takes the action
    observation, reward_c, done, info = env.step(act_taken[0])

    # Perform book-keeping
    obs = torcher(observation/255)   # Converts to float
    reward = reward_c
    episode_reward += reward
    total_reward += reward
    episode_step += 1
    total_step += 1

    # Renders the game to screen
    #env.render()

    # Add experience to replay buffer
    rp_buffer.append(obs, act_probs, act_taken_v, reward, done)
    
    # Learn from experience and clear rp buffer
    if(total_step%nsteps_to_learn == 0):
        # Calculates/Applies grads
        pl, cl, tl, dr, ce, ad = model.learn(rp_buffer)
        # Write outputs out for visualization
        tb_writer.add_scalar('Misc/CrossEntropyMean', ce.mean(), total_step) 
        tb_writer.add_scalar('Misc/Advantage', ad.mean(), total_step) 
        tb_writer.add_scalar('Loss/PolicyLoss', pl, total_step) 
        tb_writer.add_scalar('Loss/CriticLoss', cl, total_step) 
        tb_writer.add_scalar('Loss/TotalLoss', tl, total_step) 
        tb_writer.add_scalar('Rewards/DiscountedReward', dr, total_step) 
        tb_writer.add_histogram('Actions/ActionsTaken', rp_buffer.actions_scalar().cpu().numpy(), total_step, bins=np.arange(-1, env.action_space.n+1, 0.2)) 

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
        np_obs = env.reset()/255
        obs = torcher(np_obs)

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
