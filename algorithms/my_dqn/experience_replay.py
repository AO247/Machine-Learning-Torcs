# W pliku algorithms/my_dqn/experience_replay.py

from collections import deque
import random
import torch  # <-- DODANY KLUCZOWY IMPORT

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class NStepReplayMemory(ReplayMemory):
    def __init__(self, maxlen, n_step=3, gamma=0.99, device='cpu'): # <-- Dodajemy device
        super().__init__(maxlen)
        self.n_step = n_step
        self.gamma = gamma
        self.device = device # <-- Zapisujemy device
        self.buffer = deque([], maxlen=n_step)

    def append(self, transition):
        self.buffer.append(transition)

        if len(self.buffer) < self.n_step:
            return

        reward_n_step = 0
        for i in range(self.n_step):
            # transition to: (state, action, new_state, reward, terminated)
            _state, _action, _new_state, reward, _terminated = self.buffer[i]
            reward_n_step += (self.gamma ** i) * reward.item()

        state, action, _, _, _ = self.buffer[0]
        _, _, last_state, _, last_termination = self.buffer[-1]

        # Używamy self.device, które przekazaliśmy przy tworzeniu
        transition_n_step = (state, action, last_state, torch.tensor(reward_n_step, dtype=torch.float, device=self.device), last_termination)

        self.memory.append(transition_n_step)