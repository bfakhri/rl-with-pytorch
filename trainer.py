import argparse
import gym
import math
import model
import tensorflow as tf
import numpy as np

#Parsers for command line args
parser = argparse.ArgumentParser(description="Run Commands")
parser.add_argument('-t', '--prefix' , type=str, default="",
        help='Text Prepended to Tensorboard run name')
parser.add_argument('-l', '--learningRate' , type=float , default=0.0001,
        help='Learning rate of optimizer')
parser.add_argument('-m', '--maxEpisodes' , type=int, default=10000,
        help='Max Episodes to train for')
parser.add_argument('-e', '--environment_id' , type=str, default="Pong-v0",
        help='The environment to train in')
parser.add_argument('-b', '--batch_size', type=int, default=64,
        help='Number of steps to perform backprop on')
parser.add_argument('-v', '--render',type=bool, default=False,
        help='Renders gameplay to screen if true')
parser.add_argument('-c', '--cuda',type=bool, default=True,
        help='Puts all weights and ops on the GPU')
parser.add_argument('-d', '--logdir' , type=str, default="./logs",
        help='Log Directory')
parser.add_argument('-run', '--runNum' , type=str, default="",
        help='Log Directory')

# Store args in array
args = parser.parse_args()
argsArray = []
for idx, arg in enumerate(vars(args)):
    argsArray.append(arg +"="+ str(getattr(args, arg)))

# Display all hyperparams 
print("\nHyperParams:\n", argsArray, "\n")

# Turn array into string
argsString=""
for arg in argsArray:
    argsString+=arg+" "


class ReplayBuffer():
    """This class stores gameplay data as torch tensors"""

    def __init__(self, buff_len, obs_shape, act_shape):
        # Length of buffer
        self.buff_len = buff_len
        # Number of steps in the buffer
        self.n = 0
        # Observations
        self.observations = np.zeros((buff_len,)+obs_shape)
        # Action ProbabilitiesE
        self.action_probs = np.zeros((buff_len,)+(act_shape,))
        # Actions (a one-hot representation of what action was chosen)
        self.actions = np.zeros((buff_len,)+(act_shape,))
        # Rewards
        self.rewards = np.zeros(buff_len)
        # Dones 
        self.dones = np.zeros(buff_len)

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
        discounted_rewards = np.zeros(self.buff_len)
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
env = gym.make(args.environment_id)

# Instantiate the model and optimizer
model = model.Model(obs_shape=env.observation_space.shape, act_size=env.action_space.n, LR=args.learningRate, cuda=args.cuda, log_str=args.logdir+args.runNum) 

# Training Loop
episode = 0             # Current episode number
episode_step = 0        # Total steps during current episode
total_step = 0          # Total steps trained on
episode_reward = 0      # Total reward during episode
total_reward = 0        # Running count of the reward obtained
validate_freq = 20      # Number of episodes between validation phase

# Instantiate replay buffer
rp_buffer = ReplayBuffer(args.batch_size, env.observation_space.shape, env.action_space.n) 

# Gather first observation
obs = env.reset()/255.0


# Training loop
for cur_ep in range(args.maxEpisodes):
    # Asks the model for the action
    act_taken, act_taken_v, act_probs = model.act_stochastic(np.expand_dims(obs, 0))

    # Takes the action
    observation, reward_c, done, info = env.step(act_taken[0])

    # Perform book-keeping
    obs = observation/255.0   # Converts to float
    reward = reward_c
    episode_reward += reward
    total_reward += reward
    episode_step += 1
    total_step += 1

    # Renders the game to screen
    if(args.render):
        env.render()

    # Add experience to replay buffer
    rp_buffer.append(obs, act_probs, act_taken_v, reward, done)
    
    # Learn from experience and clear rp buffer
    if(total_step%args.batch_size == 0):
        # Calculates/Applies grads
        model.learn(rp_buffer)

        # Clears the replay buffer
        rp_buffer = ReplayBuffer(args.batch_size, env.observation_space.shape, env.action_space.n) 

    # Episode has finished
    if(done):
        # Update/Reset metrics
        episode += 1
        episode_step = 0
        episode_reward = 0
        np_obs = env.reset()/255
        obs = np_obs

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
