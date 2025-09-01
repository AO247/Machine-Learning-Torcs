from torcs_wrapper import GymTorcsWrapper  # <-- KLUCZOWA ZMIANA
from stable_baselines3 import SAC
import numpy as np

# Krok 1: Utwórz OPAKOWANE środowisko
env = GymTorcsWrapper(vision=False, throttle=True, gear_change=False)

# Krok 2: Załaduj wytrenowany model
# Upewnij się, że ścieżka jest poprawna z perspektywy folderu 'src'
model = SAC.load("../models/sac_torcs_agent")

# Krok 3: Pętla ewaluacyjna
print("Rozpoczynam ewaluację wytrenowanego agenta...")
for i in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Epizod {i + 1}: Ukończony w {steps} krokach. Nagroda: {total_reward:.2f}")

env.end()