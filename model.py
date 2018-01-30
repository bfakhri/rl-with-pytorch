import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, obs_shape, act_size, LR, momentum):
        # Number of possible actions
        self.act_size = act_size
        # Shape of the observations
        self.obs_shape = obs_shape
        # Dummy observation of the correct shape 
        # to do shape manipulations
        dummy_obs = np.ndarray(obs_shape)
        flat_obs = dummy_obs.reshape(-1)

        # Neural Network that defines the policy
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(36260, 50)
        self.fc_act = nn.Linear(50, act_size)
        self.fc_adv= nn.Linear(50, 1)

        # Optimizer that performs the gradient step
        self.optimizer = torch.optim.SGD(self.parameters(), lr=LR, momentum=momentum)


    def forward(self, x):
        "Takes in an observation and returns action probabilities and
        an estimate of the maximum discounted reward attainable 
        from the current state"

        x = F.relu(F.max_pool2d(self.conv1(x.permute(0,3,1,2)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 36260)
        x = F.relu(self.fc1(x))
        act = self.fc_act(x)
        act = F.softmax(act)
        adv = self.fc_adv(x)
        return act, adv


    def act_probs(self, obs):
        " Returns the action probabilities given an observation"

        obs_torch = torch.autograd.Variable(obs.unsqueeze(0))
        policy_output, reward_estimate = self.forward(obs_torch.float())
        return policy_output.data

    def learn(self, replay_buffer):
        "Performs backprop w.r.t. the replay buffer"
        
        # Clears Gradients
        self.optimizer.zero_grad()
        # Calculates the discounted reward
        discounted_reward = replay_buffer.discount(0.9)
        # Performs a foward step through the model
        policy_acts, expected_reward = self.forward(torch.autograd.Variable(replay_buffer.observations))
        # Advantage (diff b/t the actual discounted reward and the expected)
        advantage = discounted_reward - expected_reward
        # Difference between the action probabilities and the chosen action
        action_diff = torch.autograd.Variable(torch.abs(policy_acts.data - replay_buffer.actions))
        # Policy loss (encourages behavior in buffer if advantage is positive and vice-a-versa
        policy_loss = (action_diff*advantage).mean()
        # Critic loss (same as advantage) 
        critic_loss = torch.abs(advantage).mean()
        # Sums the individual losses
        total_loss = policy_loss + critic_loss
        # Calculates gradients w.r.t. all weights in the model
        total_loss.backward()
        # Applies the gradients
        self.optimizer.step()


