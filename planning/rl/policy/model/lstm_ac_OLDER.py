from planning.safe_rl.util.torch_util import to_device, to_tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# "normal" or "uniform" or None
INIT_METHOD = "normal"


def mlp(sizes, activation, output_activation=nn.Identity):
    if INIT_METHOD == "normal":
        initializer = nn.init.xavier_normal_
    elif INIT_METHOD == "uniform":
        initializer = nn.init.xavier_uniform_
    else:
        initializer = None
    bias_init = 0.0
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        if initializer is not None:
            # init layer weight
            initializer(layer.weight)
            nn.init.constant_(layer.bias, bias_init)
        layers += [layer, act()]
    return nn.Sequential(*layers)


class LSTMActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, activation, act_limit=1):
        super().__init__()


        self.act_limit = act_limit
        self.hidden_size = hidden_size

        self.head = nn.Sequential(nn.Linear(obs_dim, hidden_size), 
                                  activation())

        self.lstm = nn.LSTM(
                    input_size = hidden_size,
                    hidden_size = hidden_size,
                    num_layers = 1,
                    batch_first = True,
                    bidirectional = False,
        )

        self.decoder = nn.Sequential(nn.Linear(hidden_size, act_dim), nn.Tanh())

        self.hidden = None

    def init_hidden(self, batch_size, device):
        if(device is None):
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)

    def reset_hidden(self, batch_size, device):
        self.hidden = self.init_hidden(batch_size, device)

    def set_hidden_none(self):
        self.hidden = None

    def forward(self, obs):
        #obs: (batch, input_size)
        x = self.head(obs).unsqueeze(1) # -> (batch, seq_len =1, hidden_size)

        if self.hidden is None:
            self.hidden = self.init_hidden(batch_size=obs.size(0), device=obs.device)

        x, self.hidden = self.lstm(x, self.hidden)
        x = x[:, 0, :]

        action = self.act_limit * self.decoder(x)
        
        # Return output from network scaled to action space limits.
        return action
    
    def forward_sequential(self, obs_seq, h0 = None):
        #obs: (batch, seq_len, input_size)
        x = self.head(obs_seq)
        batch_size = obs_seq.size(0)

        # Initialize hidden state fresh for training batch
        if(h0 == None):
            hidden = self.init_hidden(batch_size, device=obs_seq.device)
        else:
            hidden = h0

        x, h_t = self.lstm(x, hidden)
        
        action = self.act_limit * self.decoder(x)
        return action, h_t
    



        


class LSTMGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_size,
                 activation):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size


        self.action_low = torch.nn.Parameter(to_tensor(action_low)[None, ...],
                                             requires_grad=False)  # (1, act_dim)
        self.action_high = torch.nn.Parameter(to_tensor(action_high)[None, ...],
                                              requires_grad=False)  # (1, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self. head = nn.Sequential(
                        nn.Linear(obs_dim, hidden_size),
                        activation()
        )

        self.lstm = nn.LSTM(
                    input_size = hidden_size,
                    hidden_size=hidden_size,
                    num_layers = 1,
                    batch_first = True
        )

        self.mu_decoder = nn.Sequential(
                    nn.Linear(hidden_size, act_dim),
                    nn.Sigmoid()
        )

        self.hidden = None


    def init_hidden(self, batch_size = 1, device =None):
        if(device is None):
            device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def set_hidden_none(self):
        self.hidden = None

    def reset_hidden(self, batch_size=1, device=None):
        self.hidden = self.init_hidden(batch_size, device)

    def _distribution(self, obs):
        x = self.head(obs)

        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if self.hidden is None or x.size(1) > 1:
            hidden = self.init_hidden(batch_size = x.size(0), device = x.device)
        else:
            hidden = self.hidden

        x , hidden = self.lstm(x, hidden)
        self.hidden = hidden

        x_last = x[:, -1, :]


        mu = self.mu_decoder(x_last)
        mu = self.action_low + (self.action_high - self.action_low) * mu
        std = torch.exp(self.log_std)
        return mu, Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1)  # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None, deterministic=False):
        '''
        Produce action distributions for given observations, and
        optionally compute the log likelihood of given actions under
        those distributions.
        If act is None, sample an action
        '''
        mu, pi = self._distribution(obs)
        if act is None:
            act = pi.sample()
        if deterministic:
            act = mu
        logp_a = self._log_prob_from_distribution(pi, act)
        return pi, act, logp_a
    

    def forward_sequential(self, obs_seq, act_seq=None, h0=None):
        x = self.head(obs_seq)
        batch_size = obs_seq.size(0)

        if h0 is None:
            hidden = self.init_hidden(batch_size, obs_seq.device)
        else:
            hidden = h0

        x, h_t = self.lstm(x, hidden)

        mu = self.mu_decoder(x)
        mu = self.action_low + (self.action_high - self.action_low) * mu
        std = torch.exp(self.log_std)
        pi = Normal(mu, std)

        if act_seq is not None:
            logp = pi.log_prob(act_seq).sum(-1)
        else:
            logp = None

        return pi, mu, logp, h_t


class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size, activation):
        super().__init__()
        self.hidden_size = hidden_size

        # Feature extractor
        self.head = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            activation()
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Value decoder
        self.value_decoder = nn.Linear(hidden_size, 1)

        # Hidden state for rolling eval
        self.hidden = None

    def init_hidden(self, batch_size, device):
        if(device is None):
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)

    def reset_hidden(self, batch_size=1, device=None):
        self.hidden = self.init_hidden(batch_size, device)

    def set_hidden_none(self):
        self.hidden = None

        
    def forward(self, obs):
        """
        obs: (batch, obs_dim) for rolling
             (batch, seq_len, obs_dim) for sequential
        """
        x = self.head(obs)
        # Add sequence dimension if rolling
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Use stored hidden state or init
        if self.hidden is None:
            #print('started')
            hidden = self.init_hidden(batch_size=x.size(0), device=x.device)
        else:
            #print('reuse')
            hidden = self.hidden

        #print(x.device)
        #print(self.lstm.weight_ih_l0.device)
        #print(hidden[0].device)
        #print(hidden[1].device)

        x, hidden = self.lstm(x, hidden)
        self.hidden = hidden

        values = self.value_decoder(x)
        return values
    
    def forward_sequential(self, obs_seq, h0 = None):
        #obs: (batch, seq_len, input_size)
        x = self.head(obs_seq)
        batch_size = obs_seq.size(0)

        # Initialize hidden state fresh for training batch
        if(h0 == None):
            hidden = self.init_hidden(batch_size, device=obs_seq.device)
        else:
            hidden = h0

        x, h_t = self.lstm(x, hidden)
        
        #x_last = x[:, -1, :]
        value = self.value_decoder(x)
        return value, h_t
