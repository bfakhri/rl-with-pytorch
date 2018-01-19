import gym
import model

#class ReplayBuffer():
#    def __init__():



# Instantiate the Environment
env = gym.make('SpaceInvaders-v0')

# Instantiate the model
model = model.Model(env.action_space.shape)

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
        print("Episode", str(episode), "finished after", episode_step, "steps with", str(episode_reward), "reward")
        episode += 1
        episode_step = 0
        episode_reward = 0
        obs = env.reset()

print("Training finished after", str(episode), "episodes and", str(total_step), "steps with avg reward", str(total_reward/episode), "reward/episode")
