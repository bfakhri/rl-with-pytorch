import torch

class ReplayBuffer():
    """This class stores gameplay data as torch tensors"""

    def __init__(self, obs_shape, act_shape, buff_len=32):
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
        self.ep_dones = np.zeros(buff_len)

    def append(self, ob, act_prob, act, r, ep_done):
        """Adds new data to buffer"""

        self.observations[self.n] = ob
        self.action_probs[self.n] = act_prob
        self.actions[self.n] = act
        self.rewards[self.n] = r
        self.ep_dones[self.n] = ep_done 
        self.n += 1

    def discount(self, lambd):
        """Returns the discounted reward parameterized by 'lambd'"""

        # There is probably a more torch-like way to do this
        discounted_rewards = np.zeros(self.buff_len)
        summer = 0
        for i in range(self.n):
            summer += math.pow(lambd, i)*self.rewards[i]
            discounted_rewards[i] = summer
            if(self.ep_dones[i] == 1):
                summer = 0
        return discounted_rewards

    def actions_scalar(self):
        """Returns array of scalars corresponding to the actions taken"""

        values, actions_s = torch.max(self.actions, dim=1)
        return actions_s
