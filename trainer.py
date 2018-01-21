import gym
import model
import numpy as np

class ReplayBuffer():
    def __init__(self, obs_shape, act_shape):
        # Number of steps in the buffer
        self.n = 0
        # Observations
        self.obs = np.empty((1,) + obs_shape, dtype=np.float)
        # Actions
        self.acts = np.empty((1,) + act_shape, dtype=np.float)
        # Rewards
        self.rs = np.empty((1,), dtype=np.float)

    def append(self, ob, act, r):
        if(self.n == 0):
            self.obs[0] = ob
            self.acts[0] = act
            self.rs[0] = r

        else:
            self.obs = np.append(self.obs, np.expand_dims(ob, 0), axis=0)
            self.acts = np.append(self.acts, np.expand_dims(act, 0), axis=0)
            self.rs = np.append(self.rs, r)

        self.n += 1

# Instantiate the Environment
env = gym.make('SpaceInvaders-v0')

# Instantiate the model
model = model.Model(env.observation_space.shape, env.action_space.n) 
# Instantiate replay buffer
rp_buffer = ReplayBuffer(env.observation_space.shape, (env.action_space.n,)) 
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
nsteps_to_learn = 40

# Training loop
while(episode < MAX_EPISODES):
    act_probs = model.act_probs(obs)
    action = np_argmax = np.argmax(act_probs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    total_reward += reward
    episode_step += 1
    total_step += 1
    env.render()
    # Add experience to replay buffer
    rp_buffer.append(obs, act_probs, reward)
    
    # Learn from experience and clear rp buffer
    if(total_step%nsteps_to_learn == 0):
        model.learn(rp_buffer)
        rp_buffer = ReplayBuffer(env.observation_space.shape, (env.action_space.n,))  

    # Episode has finished
    if(done):
        print("Episode", str(episode), "/", str(MAX_EPISODES), "finished after", episode_step, "steps with", str(episode_reward), "reward")
        episode += 1
        episode_step = 0
        episode_reward = 0
        obs = env.reset()

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
