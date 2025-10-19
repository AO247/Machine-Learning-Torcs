# algorithms/my_dqn/agent.py

import numpy as np
import torch
from torch import nn
from datetime import datetime
import os
import time

# Importujemy komponenty z naszego własnego folderu
from .experience_replay import ReplayMemory, NStepReplayMemory
from .dqn import DQN

# Importujemy klasę bazową od autora
from algorithms.common.abstract.agent import Agent as BaseAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyDQNAgent(BaseAgent):  # <-- Dziedziczymy po klasie bazowej autora!

    def __init__(self, env, args, hyper_params):
        super(MyDQNAgent, self).__init__(env, args)  # <-- Wywołujemy konstruktor rodzica

        self.hyper_params = hyper_params
        self.args = args

        # Tworzymy sieci
        self.policy_dqn = DQN(self.env.state_dim, self.env.action_dim,
                              hidden_dim=self.hyper_params['fc1_nodes'],
                              enable_dueling_dqn=self.hyper_params['enable_dueling_dqn']).to(device)
        self.target_dqn = DQN(self.env.state_dim, self.env.action_dim,
                              hidden_dim=self.hyper_params['fc1_nodes'],
                              enable_dueling_dqn=self.hyper_params['enable_dueling_dqn']).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Tworzymy optymalizator
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.hyper_params['learning_rate_a'])
        self.loss_fn = nn.MSELoss()

        # Tworzymy pamięć
        if self.hyper_params.get('enable_n_step', False):
            self.memory = NStepReplayMemory(self.hyper_params['replay_memory_size'],
                                            n_step=self.hyper_params.get('n_step', 3),
                                            gamma=self.hyper_params['discount_factor_g'],
                                            device=device)
        else:
            self.memory = ReplayMemory(self.hyper_params['replay_memory_size'])

        # Inicjalizacja zmiennych
        self.epsilon = self.hyper_params['epsilon_init']
        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0
        self.curr_state = np.zeros(self.env.state_dim)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                action = self.policy_dqn(state_tensor).argmax().item()
                return action

    def train(self):
        for self.i_episode in range(self.args.start_episode, self.args.episode_num + 1):
            is_relaunch = (self.i_episode - 1) % self.args.relaunch_period == 0
            # W `reset` przekazujemy `render` z argumentów startowych
            self.curr_state = self.env.reset(relaunch=is_relaunch, render=self.args.render, sampletrack=True)

            done = False
            score = 0
            self.episode_step = 0
            losses = []

            while not done:
                action = self.select_action(self.curr_state)
                next_state, reward, done, info = self.step(action)

                self.curr_state = next_state
                score += reward
                self.total_step += 1
                self.episode_step += 1

                if self.total_step > self.hyper_params.get("UPDATE_STARTS_FROM", 1000):
                    # Używamy TRAIN_FREQ z hiperparametrów
                    if self.total_step % self.hyper_params.get("TRAIN_FREQ", 1) == 0:
                        loss = self.update_model()
                        losses.append(loss)

            avg_loss = np.mean(losses, axis=0) if losses else (0, 0)
            self.write_log(score, avg_loss)

            self.epsilon = max(self.epsilon * self.hyper_params['epsilon_decay'], self.hyper_params['epsilon_min'])

            if self.i_episode % self.hyper_params['network_sync_rate'] == 0:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
                print("--- Target Network Synced ---")

            if self.i_episode % self.args.save_period == 0:
                print(f"--- Saving model at episode {self.i_episode} ---")
                self.save_params(self.i_episode)
            # -----------------------------------------

        self.env.close()

    def test(self):
        """Testuje wytrenowanego agenta."""
        print("--- Rozpoczynam test wytrenowanego agenta ---")

        # Wczytujemy model z podanej ścieżki
        if self.args.load_from:
            self.load_params(self.args.load_from)

        for i_episode in range(self.args.interim_test_num):
            state = self.env.reset(relaunch=True, render=self.args.render, sampletrack=True)
            done = False
            score = 0

            while not done:
                # W trybie testowym ZAWSZE wybieramy najlepszą akcję (bez epsilon)
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
                    action = self.policy_dqn(state_tensor).argmax().item()

                next_state, reward, done, info = self.env.step(action)
                state = next_state
                score += reward

            print(f"Test Epizod {i_episode + 1} | Wynik: {score:.2f}")

        self.env.close()
    def step(self, action: int) -> tuple:
        """Przekazuje akcję do środowiska i zapisuje doświadczenie."""
        next_state, reward, done, info = self.env.step(action)

        # Zapis do pamięci
        self.memory.append((
            torch.tensor(self.curr_state, dtype=torch.float, device=device),
            torch.tensor(action, dtype=torch.int64, device=device),
            torch.tensor(next_state, dtype=torch.float, device=device),
            torch.tensor(reward, dtype=torch.float, device=device),
            done
        ))

        return next_state, reward, done, info

    def write_log(self, score, avg_loss):
        """Zapisuje logi po epizodzie."""
        print(
            f"Episode {self.i_episode} | Score: {score:.2f} | Loss: {avg_loss[0]:.4f} "
            f"| Q-Value: {avg_loss[1]:.4f} | Epsilon: {self.epsilon:.3f}"
        )

    def load_params(self, path):
        """Wczytuje parametry modelu."""
        print(f"Loading model from {path}...")
        self.policy_dqn.load_state_dict(torch.load(path)["dqn_state_dict"])
        self.target_dqn.load_state_dict(torch.load(path)["dqn_state_dict"])
        print("Model loaded successfully.")

    def save_params(self, n_episode):
        """Zapisuje parametry modelu."""
        # Używamy metody save_params z klasy bazowej
        params = {
            "dqn_state_dict": self.policy_dqn.state_dict(),
        }
        super(MyDQNAgent, self).save_params(params, n_episode)

    def update_model(self):
        mini_batch = self.memory.sample(self.hyper_params['mini_batch_size'])
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            gamma = self.hyper_params['discount_factor_g'] ** self.hyper_params.get('n_step',
                                                                                    1) if self.hyper_params.get(
                'enable_n_step', False) else self.hyper_params['discount_factor_g']

            if self.hyper_params['enable_double_dqn']:
                best_actions = self.policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * gamma * self.target_dqn(new_states).gather(1,
                                                                                                     best_actions.unsqueeze(
                                                                                                         1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * gamma * self.target_dqn(new_states).max(dim=1)[0]

        current_q = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), current_q.mean().item()
