import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()

        # Zmieniamy 64 na 256
        hidden_size = 256

        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.)
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5) #Dodane *-0.5 żeby mniej gwałtowane zakręty był na początku !!!!

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)

        # Zabezpieczenie log_std przed wybuchem (clamp)
        # Zakres -20 do 2 to bezpieczny zakres dla logarytmu odchylenia standardowego
        action_logstd = torch.clamp(self.actor_logstd, -20, 2)

        action_logstd = action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # Sumujemy log_prob po wymiarach akcji (dla Continuous Env)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)