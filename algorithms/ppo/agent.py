import argparse
import os
import time
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
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


                ### potencjalna naprawa NaN ###
                with torch.no_grad():
                    is_bad = torch.isinf(ratio) | torch.isnan(ratio)

                if is_bad.any():
                    ratio = torch.where(is_bad, torch.tensor(1.0).to(device), ratio)
                ##########

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
                # newvalue = newvalue.view(-1)
                # if self.hyper_params["CLIP_VLOSS"]:
                #     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                #     v_clipped = b_values[mb_inds] + torch.clamp(
                #         newvalue - b_values[mb_inds],
                #         -self.hyper_params["CLIP_COEF"],
                #         self.hyper_params["CLIP_COEF"]
                #     )
                #     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                #     v_loss = 0.5 * v_loss_max.mean()
                # else:
                #     v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                newvalue = newvalue.view(-1)
                if self.hyper_params["CLIP_VLOSS"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # Używamy smooth_l1 zamiast potęgi **2
                    # v_loss_unclipped = F.smooth_l1_loss(newvalue, b_returns[mb_inds], reduction='none', beta=1.0)

                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.hyper_params["CLIP_COEF"],
                        self.hyper_params["CLIP_COEF"]
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    # v_loss_clipped = F.smooth_l1_loss(v_clipped, b_returns[mb_inds], reduction='none', beta=1.0)

                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    # v_loss = 0.5 * F.smooth_l1_loss(newvalue, b_returns[mb_inds], beta=1.0)

                entropy_loss = entropy.mean()
                loss = pg_loss - self.hyper_params["ENT_COEF"] * entropy_loss + v_loss * self.hyper_params["VF_COEF"]

                # --- SAFETY 4: Krytyczne zabezpieczenie przed zepsuciem modelu ---
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nCRITICAL WARNING: Loss became {loss.item()}!")
                    print(
                        f"DEBUG PARTS -> PG_Loss: {pg_loss.item()}, V_Loss: {v_loss.item()}, Entropy: {entropy_loss.item()}")

                    # Sprawdźmy wejścia
                    print(f"DEBUG INPUTS -> Max Obs: {mb_obs.max().item()}, Min Obs: {mb_obs.min().item()}")
                    print(f"DEBUG ADVANTAGES -> Max: {mb_advantages.max().item()}, Min: {mb_advantages.min().item()}")

                    print("Skipping optimization step to save the model.\n")
                    continue

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

    def write_log(self, i: int, loss: tuple, score: float = 0.0, policy_update_freq: int = 1, speed: list = None, elapsed_time: float = 0.0): # ZMIANA: Dodano elapsed_time
        """Write log about loss and score."""
        if loss is None:
            loss_arr = np.zeros(5)
        else:
            pg_loss, v_loss, entropy, approx_kl, clipfrac = loss
            loss_arr = np.array([pg_loss, approx_kl, clipfrac, v_loss, entropy])

        total_loss = loss_arr.sum()
        max_speed = 0 if speed is None or len(speed) == 0 else (max(speed))
        avg_speed = 0 if speed is None or len(speed) == 0 else (sum(speed) / len(speed))

        print(
            f"[INFO] episode {i}, step {self.episode_step}, total {self.total_step}, "
            f"score: {score:.2f}, time: {elapsed_time:.1f}s\n" # ZMIANA: Dodano wypisywanie czasu
            f"total loss: {total_loss:.3f} actor_loss: {loss_arr[0]:.3f} qf_1_loss: {loss_arr[1]:.3f} "
            f"qf_2_loss: {loss_arr[2]:.3f} vf_loss: {loss_arr[3]:.3f} alpha_loss: {loss_arr[4]:.3f}\n"
            f"track name: {self.env.track_name}, race position: {self.env.last_obs['racePos']}, "
            f"max speed {max_speed:.2f}, avg speed {avg_speed:.2f}\n"
        )

        if self.args.log:
            with open(self.log_filename, "a") as file:
                file.write(
                    "%d;%d;%d;%d;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%s;%d;%.2f;%.2f;%.2f\n" # ZMIANA: Dodano %.2f na końcu
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
                        avg_speed,
                        elapsed_time # ZMIANA: Przekazanie czasu do pliku
                    )
                )

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
        """Main training loop with Lag-Free Updates."""

        # --- KONFIGURACJA ---
        num_steps_target = self.hyper_params["NUM_STEPS"]
        max_episodes = self.args.episode_num
        self.hyper_params["BATCH_SIZE"] = int(num_steps_target)
        self.hyper_params["MINIBATCH_SIZE"] = int(
            self.hyper_params["BATCH_SIZE"] // self.hyper_params["NUM_MINIBATCHES"])

        # --- WZNOWIENIE ---
        start_time_session = time.time()
        time_offset = 0.0

        if self.args.start_episode > 1:
            file_mode = "a"
            self.total_step, time_offset = self._recover_total_step()
            if self.total_step == 0:
                print("[WARNING] Could not recover step. Starting from 0.")
        else:
            file_mode = "w"
            self.total_step = 0

        if self.args.log:
            with open(self.log_filename, file_mode) as file:
                if self.args.start_episode == 1:
                    file.write(str(self.args) + "\n")
                    file.write(str(self.hyper_params) + "\n")
                else:
                    file.write(f"\n--- RESUMING FROM EPISODE {self.args.start_episode} ---\n")

        # --- INICJALIZACJA ---
        self.episode_step = 0
        current_episode = self.args.start_episode
        last_loss_tuple = None

        mem_obs, mem_actions, mem_logprobs = [], [], []
        mem_rewards, mem_dones, mem_values = [], [], []

        # Pierwszy reset na start
        next_obs = torch.Tensor(self.env.reset(relaunch=True, sampletrack=True)).to(device).view(1, -1)
        next_done = torch.zeros(1).to(device)

        current_ep_return = 0
        current_ep_speeds = []

        print("Starting PPO training loop (Sync Fix)...")

        # GŁÓWNA PĘTLA
        while current_episode <= max_episodes:

            if self.hyper_params["ANNEAL_LR"]:
                frac = 1.0 - (current_episode / max_episodes)
                lrnow = max(frac * self.hyper_params["LEARNING_RATE"], 0.0)
                self.optimizer.param_groups[0]["lr"] = lrnow

            # Flag to track if we need to update model
            perform_update = False

            # Pętla kroków (Rollout)
            while True:
                self.total_step += 1
                self.episode_step += 1

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)

                real_action = action.cpu().numpy().flatten()
                next_obs_np, reward, done = self.step(real_action)

                # Zapis do pamięci
                mem_obs.append(next_obs)
                mem_actions.append(action)
                mem_logprobs.append(logprob)
                mem_values.append(value.flatten())

                # Clip nagrody (bez dzielenia, tylko ucięcie szpilek)
                train_reward = np.clip(reward, -20.0, 20.0)
                mem_rewards.append(torch.tensor(train_reward).to(device).view(-1))
                mem_dones.append(torch.tensor(done).float().to(device).view(-1))

                next_obs = torch.Tensor(next_obs_np).to(device).view(1, -1)

                current_ep_return += reward
                current_ep_speeds.append(self.env.last_speed)

                if done:
                    # Logowanie
                    current_time = time_offset + (time.time() - start_time_session)
                    self.write_log(current_episode, last_loss_tuple, current_ep_return, speed=current_ep_speeds,
                                   elapsed_time=current_time)

                    if current_episode % self.args.save_period == 0:
                        self.save_params(current_episode)

                    current_episode += 1
                    current_ep_return = 0
                    current_ep_speeds = []
                    self.episode_step = 0

                    # --- KLUCZOWA ZMIANA LOGIKI ---
                    # Sprawdzamy czy mamy dość danych do update'u
                    if len(mem_obs) >= num_steps_target:
                        perform_update = True
                        break  # Wychodzimy z pętli kroków, ALE NIE RESETUJEMY JESZCZE GRY!
                    else:
                        # Jeśli nie robimy update'u, to resetujemy grę natychmiast i gramy dalej
                        is_relaunch = (current_episode % self.args.relaunch_period == 0)
                        raw_obs = self.env.reset(relaunch=is_relaunch, sampletrack=True)
                        next_obs = torch.Tensor(raw_obs).to(device).view(1, -1)

                    if current_episode > max_episodes:
                        break

            if current_episode > max_episodes:
                break

            # --- SEKCJA UPDATE (Tylko gdy flaga perform_update jest True) ---
            if perform_update:

                # 1. Przygotowanie danych
                b_obs = torch.cat(mem_obs)
                b_actions = torch.cat(mem_actions)
                b_logprobs = torch.cat(mem_logprobs)
                b_rewards = torch.cat(mem_rewards)
                b_dones = torch.cat(mem_dones)
                b_values = torch.cat(mem_values)

                current_batch_size = b_obs.shape[0]
                self.hyper_params["BATCH_SIZE"] = current_batch_size
                self.hyper_params["MINIBATCH_SIZE"] = int(current_batch_size // self.hyper_params["NUM_MINIBATCHES"])

                # 2. Obliczenie GAE
                with torch.no_grad():
                    next_value = 0  # Bo update jest zawsze po done
                    if self.hyper_params["GAE"]:
                        advantages = torch.zeros_like(b_rewards).to(device)
                        lastgaelam = 0
                        for t in reversed(range(current_batch_size)):
                            if t == current_batch_size - 1:
                                nextnonterminal = 0.0
                                nextvalues = next_value
                            else:
                                nextnonterminal = 1.0 - b_dones[t]
                                nextvalues = b_values[t + 1]
                            delta = b_rewards[t] + self.hyper_params["GAMMA"] * nextvalues * nextnonterminal - b_values[
                                t]
                            advantages[t] = lastgaelam = delta + self.hyper_params["GAMMA"] * self.hyper_params[
                                "GAE_LAMBDA"] * nextnonterminal * lastgaelam
                        returns = advantages + b_values
                    else:
                        returns = torch.zeros_like(b_rewards).to(device)
                        for t in reversed(range(current_batch_size)):
                            if t == current_batch_size - 1:
                                nextnonterminal = 0.0
                                next_return = 0.0
                            else:
                                nextnonterminal = 1.0 - b_dones[t]
                                next_return = returns[t + 1]
                            returns[t] = b_rewards[t] + self.hyper_params["GAMMA"] * nextnonterminal * next_return
                        advantages = returns - b_values

                # 3. Właściwa aktualizacja (Tu następuje "Lag")
                # Ponieważ wyszliśmy z pętli kroków po 'done', gra i tak czeka na reset.
                # Ten czas obliczeń nie wpłynie na jazdę.

                b_obs = b_obs.reshape((-1, self.env.state_dim))
                b_actions = b_actions.reshape((-1, self.env.action_dim))

                last_loss_tuple = self.update_model(b_obs, b_actions, b_logprobs, b_returns=returns,
                                                    b_advantages=advantages, b_values=b_values)

                # 4. Czyszczenie pamięci
                mem_obs.clear()
                mem_actions.clear()
                mem_logprobs.clear()
                mem_rewards.clear()
                mem_dones.clear()
                mem_values.clear()

                # --- RESET PO UPDATE (Teraz jest bezpiecznie) ---
                # Dopiero teraz, po zakończeniu mielenia cyfr, resetujemy grę.
                # Dzięki temu gra startuje i od razu dostaje sterowanie, bez laga.

                # Opcjonalnie: Po ciężkich obliczeniach można wymusić relaunch dla świeżości procesu
                is_relaunch = (current_episode % self.args.relaunch_period == 0)

                raw_obs = self.env.reset(relaunch=is_relaunch, sampletrack=True)
                next_obs = torch.Tensor(raw_obs).to(device).view(1, -1)
                # ------------------------------------------------

        self.env.close()
        self.writer.close()
        self.save_params(current_episode)

    def _recover_total_step(self):
        """Próbuje odczytać ostatni total_step i czas z pliku logów."""
        if not os.path.exists(self.log_filename):
            return 0, 0.0  # ZMIANA: Zwraca krotkę

        try:
            with open(self.log_filename, "r") as f:
                lines = f.readlines()

            for line in reversed(lines):
                line = line.strip()
                if not line or line.startswith(("Namespace", "{", "---", "[INFO]", "Ratio")):
                    continue

                parts = line.split(";")
                if len(parts) >= 3:
                    recovered_step = int(parts[2])

                    # ZMIANA: Próba odczytu czasu z ostatniej kolumny
                    recovered_time = 0.0
                    try:
                        # Sprawdzamy czy ostatni element to liczba (czas)
                        last_val = parts[-1]
                        if "Ratio" not in last_val:
                            recovered_time = float(last_val)
                    except ValueError:
                        pass  # Stare logi nie mają czasu

                    print(f"[INFO] Recovered total_step: {recovered_step}, time: {recovered_time}")
                    return recovered_step, recovered_time  # ZMIANA: Zwraca krotkę

            return 0, 0.0

        except Exception as e:
            print(f"[WARNING] Failed to recover info from logs: {e}. Starting from 0.")
            return 0, 0.0