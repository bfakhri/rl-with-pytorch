import numpy as np
import torch

class Model():
    def __init__(self, obs_shape, act_size):
        # Number of possible actions
        self.act_size = act_size
        # Shape of the observations
        self.obs_shape = obs_shape
        # Dummy observation of the correct shape 
        # to do shape manipulations
        dummy_obs = np.ndarray(obs_shape)
        flat_obs = dummy_obs.reshape(-1)

        # Define a single layer for the action
        self.policy = torch.nn.Linear(len(flat_obs), act_size)


    def act(self, obs):
        " Right now it is just a random agent"
        obs_torch = torch.autograd.Variable(torch.Tensor.float(torch.from_numpy(obs.reshape(-1))))
        policy_output = self.policy(obs_torch)
        np_argmax = np.argmax(policy_output.data.numpy())
        return np_argmax

    def learn(self, replay_buffer):
        "Performs backprop w.r.t. the replay buffer"
        return 0


