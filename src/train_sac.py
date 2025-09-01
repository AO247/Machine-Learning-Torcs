from torcs_wrapper import GymTorcsWrapper # <-- ZMIANA TUTAJ
from stable_baselines3 import SAC
import numpy as np

# Krok 1: Utwórz opakowane środowisko TORCS
env = GymTorcsWrapper(vision=False, throttle=True, gear_change=False)

# Krok 2: Zdefiniuj model SAC
model = SAC("MlpPolicy", env, verbose=1)

# Krok 3: Rozpocznij trening!
print("Rozpoczynam trening agenta SAC...")
model.learn(total_timesteps=1000)
print("Trening zakończony.")

# Krok 4: Zapisz wytrenowany model (ścieżka jest teraz poprawna)
model.save("../models/sac_torcs_agent")
print("Model zapisany w folderze '../models/'.")

# Krok 5: Zamknij środowisko
env.end()