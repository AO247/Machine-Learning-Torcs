import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch

from algorithms.ppo.agent import PPO


def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64x4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9   # alpha of learning rate
    discount_factor_g = 0.9 # gamma or discount factor

    epsilon = 1                     # 1 = 100% random actions
    epsilon_decay_rate = 0.00005     # epsilon decay rate
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner, 63= bottom right corner
        terminated = False      # True when fall in the hole
        truncated = False       # True when action > 200

        actor(state)

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action])

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1
        if(i % 1000 == 0):
            print("Epsilon: " + str(epsilon))
            print("Reward: " + str(reward))

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()


if __name__ == '__main__':

    env = gym.make('FrozenLake-v1', map_name="8x8")

    state_dimension = env.observation_space.n
    action_dimension = env.action_space.n

    print(f"Rozmiar wejscia (stanu): {state_dimension}")
    print(f"Rozmiar wejscia (akcji): {action_dimension}")

    ppo_network = PPO(state_dimension, action_dimension)
    print(f"\Architektura sieci: ")
    print(ppo_network)

    current_state_id = 10

    dummy_state = torch.zeros(state_dimension)

    dummy_state[current_state_id] = 1.0

    with torch.no_grad():
        action_distribution, state_value = ppo_network(dummy_state)

    print(f"\nWynik dla stanu {current_state_id}:")
    print(f"  Oszacowana wartość stanu (Krytyk): {state_value.item():.4f}")
    print(f"  Prawdopodobieństwa akcji (Aktor):")
    action_labels = ['Lewo', 'Dół', 'Prawo', 'Góra']
    for i, prob in enumerate(action_distribution.probs):
        print(f"    - {action_labels[i]}: {prob.item():.2%}")

    # Możemy też wylosować akcję z tej dystrybucji
    sampled_action = action_distribution.sample()
    print(f"\n  Wylosowana akcja: {sampled_action.item()} ({action_labels[sampled_action.item()]})")

