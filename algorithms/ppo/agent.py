import argparse
import os
import time
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algorithms.common.abstract.agent import Agent
from env.torcs_envs import DefaultEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent(Agent):
    def __init__(
            self,
            env: DefaultEnv,
            args: argparse.Namespace,
            hyper_params: dict,
            model: nn.Module,
            optimizer: optim.Optimizer,
    ):
        """Initialization."""
        super().__init__(env, args)

        self.agent = model
        self.optimizer = optimizer
        self.hyper_params = hyper_params

        # Tensorboard writer
        self.writer = SummaryWriter(f"runs/{args.algo}_{args.seed}_{int(time.time())}")

        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)

        with torch.no_grad():
            if self.args.test:
                # --- TRYB TESTOWY: DETERMINISTYCZNY ---
                # Nie losujemy! Bierzemy "średnią" z rozkładu (to, co sieć uważa za najlepsze)
                # Musimy odwołać się bezpośrednio do sieci actor_mean z modelu
                action = self.agent.actor_mean(state_tensor)
            else:
                # --- TRYB TRENINGOWY: STOCHASTYCZNY ---
                # Losujemy z rozkładu (eksploracja)
                action, _, _, _ = self.agent.get_action_and_value(state_tensor)

        return action.cpu().numpy().flatten()

    def step(self, action: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        # Jeśli akcja jest tensorem, zamieniamy na numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().flatten()

        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def update_model(self, b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values) -> Tuple[float, ...]:
        """Train the model using collected batch with NaN safety checks."""

        batch_size = self.hyper_params["BATCH_SIZE"]
        minibatch_size = self.hyper_params["MINIBATCH_SIZE"]

        b_inds = np.arange(batch_size)
        clipfracs = []

        # Statystyki do logowania
        avg_pg_loss = 0
        avg_v_loss = 0
        avg_entropy_loss = 0
        avg_approx_kl = 0
        valid_updates = 0  # Licznik udanych aktualizacji

        # Optimization Loop
        for epoch in range(self.hyper_params["UPDATE_EPOCHS"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # --- SAFETY 1: Sanityzacja wejścia (b_obs) ---
                mb_obs = b_obs[mb_inds]
                if torch.isnan(mb_obs).any():
                    # print("WARNING: NaN detected in batch observations! Replacing with zeros.")
                    mb_obs = torch.where(torch.isnan(mb_obs), torch.zeros_like(mb_obs), mb_obs)

                # --- SAFETY 2: Sanityzacja akcji (b_actions) ---
                mb_actions = b_actions[mb_inds]
                if torch.isnan(mb_actions).any():
                    mb_actions = torch.where(torch.isnan(mb_actions), torch.zeros_like(mb_actions), mb_actions)

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    mb_obs, mb_actions
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.hyper_params["CLIP_COEF"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # --- SAFETY 3: Sanityzacja advantages ---
                if torch.isnan(mb_advantages).any():
                    mb_advantages = torch.where(torch.isnan(mb_advantages), torch.zeros_like(mb_advantages),
                                                mb_advantages)

                if self.hyper_params["NORM_ADV"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.hyper_params["CLIP_COEF"],
                                                        1 + self.hyper_params["CLIP_COEF"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.hyper_params["CLIP_VLOSS"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.hyper_params["CLIP_COEF"],
                        self.hyper_params["CLIP_COEF"]
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.hyper_params["ENT_COEF"] * entropy_loss + v_loss * self.hyper_params["VF_COEF"]

                # --- SAFETY 4: Krytyczne zabezpieczenie przed zepsuciem modelu ---
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"CRITICAL WARNING: Loss is {loss.item()}! Skipping optimization step to save the model.")
                    continue  # Pomijamy ten krok, nie robimy optimizer.step()!

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.hyper_params["MAX_GRAD_NORM"])
                self.optimizer.step()

                # Akumulacja do średniej (tylko poprawne kroki)
                avg_pg_loss += pg_loss.item()
                avg_v_loss += v_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_approx_kl += approx_kl.item()
                valid_updates += 1

            if self.hyper_params["TARGET_KL"] is not None:
                if approx_kl > self.hyper_params["TARGET_KL"]:
                    break

        # Obliczanie średnich strat (zabezpieczenie przez dzieleniem przez 0)
        if valid_updates == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        return (
            avg_pg_loss / valid_updates,
            avg_v_loss / valid_updates,
            avg_entropy_loss / valid_updates,
            avg_approx_kl / valid_updates,
            np.mean(clipfracs) if clipfracs else 0.0
        )

    def write_log(self, i: int, loss: tuple, score: float = 0.0, policy_update_freq: int = 1, speed: list = None):
        """
        Write log about loss and score.
        Format compatible with SAC implementation for easy parsing.
        """
        # 1. Obsługa sytuacji, gdy loss nie został jeszcze policzony (pierwszy epizod)
        if loss is None:
            loss_arr = np.zeros(5)
        else:
            # Rozpakowanie krotki z PPO
            pg_loss, v_loss, entropy, approx_kl, clipfrac = loss

            # 2. MAPOWANIE PPO -> Format SAC (5 elementów)
            # SAC ma: [Actor, QF1, QF2, VF, Alpha]
            # My mapujemy:
            # [0] Actor Loss  -> Policy Loss
            # [1] QF1 Loss    -> Approx KL (Warto to śledzić, wstawiamy w slot QF1)
            # [2] QF2 Loss    -> Clip Frac (Wstawiamy w slot QF2)
            # [3] VF Loss     -> Value Loss
            # [4] Alpha Loss  -> Entropy Loss
            loss_arr = np.array([pg_loss, approx_kl, clipfrac, v_loss, entropy])

        # Suma (w PPO to nie jest prawdziwy 'total loss', ale zachowujemy format)
        total_loss = loss_arr.sum()

        max_speed = 0 if speed is None or len(speed) == 0 else (max(speed))
        avg_speed = 0 if speed is None or len(speed) == 0 else (sum(speed) / len(speed))

        # Wydruk do konsoli (nazwy zmiennych będą z SAC, ale wartości z PPO - patrz mapowanie wyżej)
        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss(pg): %.3f qf_1_loss(kl): %.3f qf_2_loss(clip): %.3f "
            "vf_loss: %.3f alpha_loss(ent): %.3f\n"
            "track name: %s, race position: %d, max speed %.2f, avg speed %.2f\n"
            % (
                i,
                self.episode_step,  # Musimy to aktualizować w train()!
                self.total_step,  # Musimy to aktualizować w train()!
                score,
                total_loss,
                loss_arr[0] * policy_update_freq,
                loss_arr[1],
                loss_arr[2],
                loss_arr[3],
                loss_arr[4],
                self.env.track_name,
                self.env.last_obs['racePos'],
                max_speed,
                avg_speed
            )
        )

        # Zapis do pliku
        if self.args.log:
            with open(self.log_filename, "a") as file:
                file.write(
                    "%d;%d;%d;%d;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%s;%d;%.2f;%.2f\n"
                    % (
                        i,
                        self.episode_step,
                        self.total_step,
                        score,
                        total_loss,
                        loss_arr[0] * policy_update_freq,
                        loss_arr[1],
                        loss_arr[2],
                        loss_arr[3],
                        loss_arr[4],
                        self.env.track_name,
                        self.env.last_obs['racePos'],
                        max_speed,
                        avg_speed
                    )
                )

        # Opcjonalnie: Zapis do Tensorboarda (już z poprawnymi nazwami PPO)
        if loss is not None:
            self.writer.add_scalar("charts/episodic_return", score, i)
            self.writer.add_scalar("losses/policy_loss", pg_loss, i)
            self.writer.add_scalar("losses/value_loss", v_loss, i)
            self.writer.add_scalar("losses/entropy", entropy, i)
            self.writer.add_scalar("losses/approx_kl", approx_kl, i)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path, map_location=device)
        self.agent.load_state_dict(params["model"])
        self.optimizer.load_state_dict(params["optimizer"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "model": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        super().save_params(params, n_episode)

    def train(self):
        """Main training loop."""
        self.total_step = 0
        self.episode_step = 0

        # Konfiguracja
        num_steps = self.hyper_params["NUM_STEPS"]
        max_episodes = self.args.episode_num

        self.hyper_params["BATCH_SIZE"] = int(num_steps)
        self.hyper_params["MINIBATCH_SIZE"] = int(
            self.hyper_params["BATCH_SIZE"] // self.hyper_params["NUM_MINIBATCHES"])

        # Alokacja buforów
        obs = torch.zeros((num_steps, self.env.state_dim)).to(device)
        actions = torch.zeros((num_steps, self.env.action_dim)).to(device)
        logprobs = torch.zeros((num_steps)).to(device)
        rewards = torch.zeros((num_steps)).to(device)
        dones = torch.zeros((num_steps)).to(device)
        values = torch.zeros((num_steps)).to(device)

        current_episode = 1
        last_loss_tuple = None
        global_step = 0

        # Reset środowiska
        next_obs = torch.Tensor(self.env.reset(relaunch=True, sampletrack=True)).to(device).view(1, -1)
        next_done = torch.zeros(1).to(device)

        current_ep_return = 0
        current_ep_speeds = []

        print("Starting PPO training loop...")

        while current_episode <= max_episodes:

            if self.hyper_params["ANNEAL_LR"]:
                frac = 1.0 - (current_episode / max_episodes)
                lrnow = max(frac * self.hyper_params["LEARNING_RATE"], 0.0)
                self.optimizer.param_groups[0]["lr"] = lrnow

            # --- ROLLOUT PHASE ---
            for step in range(0, num_steps):
                self.total_step += 1
                self.episode_step += 1
                global_step += 1

                # --- SANITIZATION (Naprawa błędu NaN z TORCS) ---
                if torch.isnan(next_obs).any():
                    next_obs = torch.zeros_like(next_obs)
                # ----------------------------------------

                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                real_action = action.cpu().numpy().flatten()
                next_obs_np, reward, done = self.step(real_action)

                # reward = reward / 10.0
                # reward = np.clip(reward, -20.0, 20.0)

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs_np).to(device).view(1, -1)
                next_done = torch.Tensor([done]).to(device)

                current_ep_return += reward
                current_ep_speeds.append(self.env.last_speed)

                if done:
                    # Logowanie wyników epizodu
                    self.write_log(current_episode, last_loss_tuple, current_ep_return, speed=current_ep_speeds)

                    # Zapis modelu (zrobione tutaj, więc nie trzeba na dole pętli)
                    if current_episode % self.args.save_period == 0:
                        self.save_params(current_episode)

                    current_episode += 1
                    current_ep_return = 0
                    current_ep_speeds = []
                    self.episode_step = 0

                    is_relaunch = (current_episode % self.args.relaunch_period == 0) and (step > num_steps - 50)

                    # Pobieramy nowy stan
                    raw_obs = self.env.reset(relaunch=is_relaunch, sampletrack=True)
                    next_obs = torch.Tensor(raw_obs).to(device).view(1, -1)

                    if current_episode > max_episodes:
                        break

            if current_episode > max_episodes:
                break

            # --- GAE ---
            with torch.no_grad():
                if torch.isnan(next_obs).any():
                    next_obs = torch.zeros_like(next_obs)

                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                if self.hyper_params["GAE"]:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + self.hyper_params["GAMMA"] * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + self.hyper_params["GAMMA"] * self.hyper_params[
                            "GAE_LAMBDA"] * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + self.hyper_params["GAMMA"] * nextnonterminal * next_return
                    advantages = returns - values

            b_obs = obs.reshape((-1, self.env.state_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, self.env.action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # --- UPDATE PHASE ---
            last_loss_tuple = self.update_model(b_obs, b_actions, b_logprobs, b_returns, b_advantages, b_values)

            # USUNIĘTO: Błędny blok z update_count

        self.env.close()
        self.writer.close()
        self.save_params(current_episode)