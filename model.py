import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, obs_shape, act_size):
        # Number of possible actions
        self.act_size = act_size
        # Shape of the observations
        self.obs_shape = obs_shape
        # Dummy observation of the correct shape 
        # to do shape manipulations
        dummy_obs = np.ndarray(obs_shape)
        flat_obs = dummy_obs.reshape(-1)
        # Neural Network that defines the policy
        #nn_out = torch.nn.Linear(len(flat_obs), act_size)
        #self.policy = torch.nn.functional.softmax(nn_out) 

        # Neural Network that defines the policy
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(36260, 50)
        self.fc2 = nn.Linear(50, act_size)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x.permute(0,3,1,2)), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x 



    def act_probs(self, obs):
        " Right now it is just a random agent"
        #obs_torch = torch.autograd.Variable(torch.Tensor.float(torch.from_numpy(obs.reshape(-1))))
        obs_torch = torch.autograd.Variable(torch.Tensor.float(torch.from_numpy(obs).unsqueeze(0)))
        policy_output = self.forward(obs_torch).data.numpy()
        return policy_output 

    def learn(self, replay_buffer):
        "Performs backprop w.r.t. the replay buffer"
        discounted_reward = replay_buffer.discount(0.9)
        print(discounted_reward)
        return 0


