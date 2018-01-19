import numpy as np
import torch

class model():
    def __init__(self, act_size):
        self.act_size = act_size

    def act(self, obs):
        " Right now it is just a random agent"
        action = np.zeros(self.act_size, dtype=np.int)
        act_idx = np.random.randint(0, len(action))
        action[act_idx] = 1
        return action

    def learn(self, replay_buffer):
        "Performs backprop w.r.t. the replay buffer"
        return 0


m = model((5))
for i in range(10):
    print(m.act((0)))

