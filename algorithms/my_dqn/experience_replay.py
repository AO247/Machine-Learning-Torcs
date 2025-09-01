# Define memory for Experience Replay
from collections import deque
import random
class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        # Optional seed for reproducibility
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)




class NStepReplayMemory(ReplayMemory):
    def __init__(self, maxlen, n_step=3, gamma=0.99):
        super().__init__(maxlen)
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque([], maxlen=n_step)

    def append(self, transition):
        # Dodaj nowe doświadczenie do tymczasowego bufora
        self.buffer.append(transition)

        # Jeśli bufor nie jest jeszcze pełny, nic więcej nie rób
        if len(self.buffer) < self.n_step:
            return

        # Jeśli bufor jest pełny, oblicz nagrodę N-krokową i stan N-krokowy
        # R = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        reward_n_step = 0
        for i in range(self.n_step):
            reward_n_step += (self.gamma ** i) * self.buffer[i][3].item() # buffer[i][3] to nagroda

        # Pierwsze doświadczenie w buforze
        state, action, _, _, _ = self.buffer[0]
        # Ostatnie doświadczenie w buforze
        _, _, last_state, _, last_termination = self.buffer[-1]

        # Tworzymy nowe, "skompresowane" doświadczenie N-krokowe
        transition_n_step = (state, action, last_state, torch.tensor(reward_n_step, dtype=torch.float, device=self.memory[0][0].device), last_termination)

        # Dodajemy je do głównej pamięci
        self.memory.append(transition_n_step)