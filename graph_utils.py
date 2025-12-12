import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import string

## Various algorithms log order per column
a2c_columns = {"episode": 0, "score": 1, "total_loss": 2, "policy_loss": 3, "value_loss": 4}
per_ddpg_columns = {"episode": 0, "episode_step": 1, "total_step": 2, "total_score": 3, "total_loss": 4,
                    "actor_loss": 5, "critic_loss": 6}

dqn_columns = {
    "episode": 0,
    "episode_step": 1,
    "total_step": 2,
    "total_score": 3,
    "epsilon": 4,
    "total_loss": 5,
    "avg_q_value": 6,
    "avg_time_cost": 7,
    "track_name": 8,
    "race position": 9,
    "max_speed": 10,
    "avg_speed": 11
}

sac_columns = {
    "episode": 0,
    "episode_step": 1,
    "total_step": 2,
    "total_score": 3,
    "total_loss": 4,
    "actor_loss": 5,
    "qf_1_loss": 6,
    "qf_2_loss": 7,
    "vf_loss": 8,
    "alpha_loss": 9,
    "track_name": 10,
    "race_position": 11,
    "max_speed": 12,
    "avg_speed": 13
}
ppo_columns = {
    "episode": 0,
    "episode_step": 1,
    "total_step": 2,
    "total_score": 3,
    "total_loss": 4,
    "actor_loss": 5,  # Policy Loss
    "approx_kl": 6,  # W miejscu QF1
    "clip_frac": 7,  # W miejscu QF2
    "vf_loss": 8,  # Value Loss
    "entropy": 9,  # W miejscu Alpha
    "track_name": 10,
    "race_position": 11,
    "max_speed": 12,
    "avg_speed": 13
}

algo_column_list = {
    "dqn": dqn_columns,
    "DQN": dqn_columns,
    "sac": sac_columns,
    "sac-lstm": sac_columns, # SAC-LSTM ma ten sam format co SAC
    "SAC": sac_columns,
    "SACLSTM": sac_columns,
    "ppo": ppo_columns,
    "PPO": ppo_columns
}

# ------------------------------------------------------------------

t3d_columns = {"episode": 0, "episode_step": 1, "total_step": 2, "total_score": 3, "total_loss": 4, "actor:loss": 5,
               "critic1_loss": 6, "critic2_loss": 7, "track_name": 8, "race_position": 9, "max_speed": 10,
               "avg_speed": 11}

algo_column_list = {
    "a2c": a2c_columns,
    "per-ddpg": per_ddpg_columns,
    "dqn": dqn_columns,
    "sac": sac_columns,
    "SAC": sac_columns,
    "SACLSTM": sac_columns,
    "sac-lstm": sac_columns,
    "t3d": t3d_columns,
    # --- DODANO PPO ---
    "ppo": ppo_columns,
    "PPO": ppo_columns
    # ------------------
}


def get_column_indice(algo, column_name):
    # Obsługa specjalnych nazw kolumn dla różnych algorytmów
    if algo == "ppo":
        if column_name == "alpha_loss": return ppo_columns["entropy"]  # Mapowanie PPO Entropy na Alpha

    # Ujednolicenie nazw algorytmów (np. sac-lstm -> sac)
    if "sac" in algo.lower():
        algo_key = "sac"
    elif "dqn" in algo.lower():
        algo_key = "dqn"
    elif "ppo" in algo.lower():
        algo_key = "ppo"
    else:
        algo_key = algo

    if algo_key in algo_column_list:
        columns = algo_column_list[algo_key]
        if column_name in columns:
            return columns[column_name]

    print(f"[Warning] Column '{column_name}' not found for algo '{algo}'")
    return -1


def get_color(color_index):
    colors = ["red", "blue", "green", "magenta", "orange", "black", "cyan", "purple", "brown"]
    return colors[color_index % len(colors)]


def smoother(array, ws):
    """ Return smoothed array by the mean filter """
    if len(array) == 0: return np.array([])
    if len(array) < ws: return np.array(array)
    return np.array([sum(array[i:i + ws]) / ws for i in range(len(array) - ws)])


def read_log_file(filename, x_indice, y_indice):
    x_values = []
    y_values = []

    if not os.path.exists(filename):
        print(f"[Error] File not found: {filename}")
        return [], []

    with open(filename, "r") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("Namespace") or line.startswith("{") or line.startswith(
                "---") or line.startswith("[INFO]"):
            continue

        try:
            tokens = line.split(";")
            if len(tokens) <= max(x_indice, y_indice):
                continue

            if x_indice == -1 or y_indice == -1: continue

            x_value = int(tokens[x_indice])
            y_value = float(tokens[y_indice])
            x_values.append(x_value)
            y_values.append(y_value)
        except ValueError:
            continue

    return x_values, y_values


def persist_figure(plt, title):
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    rnd_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    # Nazwa pliku zawiera prefix algorytmu przekazany w tytule
    filename = f"graphs/{title}_{rnd_filename}.jpg"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved graph: {filename}")


def plot_multi_algo_single_feature(files, x_column, y_column, smooth_factor=100, prefix=""):
    color_index = 0
    plt.figure(figsize=(10, 6))

    has_data = False

    for file in files:
        basename = os.path.basename(file)
        try:
            # Próba wyciągnięcia nazwy algorytmu z pliku (np. Torcs_ppo_...)
            algo = basename.split("_")[1]
        except:
            algo = "unknown"

        x_indice = get_column_indice(algo, x_column)
        y_indice = get_column_indice(algo, y_column)

        x_vals, y_vals = read_log_file(file, x_indice, y_indice)

        if not x_vals:
            continue

        # Smooth
        y_vals = smoother(y_vals, smooth_factor)
        x_vals = x_vals[:len(y_vals)]

        if len(x_vals) > 0:
            color = get_color(color_index)
            color_index += 1
            # Etykieta to nazwa algorytmu (lub pliku, jeśli chcesz dokładniej)
            label_text = f"{algo.upper()}"
            plt.plot(x_vals, y_vals, color=color, label=label_text, linewidth=1.5)
            has_data = True

    if not has_data:
        print(f"No data found for {y_column} vs {x_column}")
        plt.close()
        return

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{prefix}{y_column} vs {x_column} (Smooth: {smooth_factor})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Zapis z prefixem w nazwie pliku
    persist_figure(plt, f"{prefix}{y_column}_vs_{x_column}")
    # plt.show() # Odkomentuj, jeśli chcesz widzieć okna
    plt.close()


# --- FUNKCJE DEDYKOWANE ---

def generate_ppo_plots(log_files, smoothing=100):
    print("\n=== Generowanie wykresów dla PPO ===")
    prefix = "PPO_"
    # Standardowe metryki
    plot_multi_algo_single_feature(log_files, "episode", "total_score", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "avg_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "max_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "alpha_loss", smoothing, prefix)  # Entropy


def generate_dqn_plots(log_files, smoothing=100):
    print("\n=== Generowanie wykresów dla DQN ===")
    prefix = "DQN_"
    # DQN ma epsilon
    plot_multi_algo_single_feature(log_files, "episode", "total_score", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "avg_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "max_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "total_loss", smoothing, prefix)


def generate_sac_plots(log_files, smoothing=100):
    print("\n=== Generowanie wykresów dla SAC-LSTM ===")
    prefix = "SAC-LSTM_"
    # SAC ma alpha_loss
    plot_multi_algo_single_feature(log_files, "episode", "total_score", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "avg_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "max_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "alpha_loss", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "total_loss", smoothing, prefix)


# --- MAIN ---
if __name__ == "__main__":

    # --- TUTAJ WPISZ ŚCIEŻKI DO SWOICH PLIKÓW ---

    path_ppo = "logs/Torcs_ppo_3bbd3b1.txt"  # <-- Twoja ścieżka PPO
    path_dqn = "logs/Torcs_dqn_3074557.txt"  # <-- Twoja ścieżka DQN
    path_sac = "logs/seven_cars/Torcs_sac-lstm_84389f4_fin.txt"  # <-- Twoja ścieżka SAC

    smoothing_window = 100

    # 1. PPO
    if os.path.exists(path_ppo):
        generate_ppo_plots([path_ppo], smoothing=smoothing_window)
    else:
        print(f"Brak pliku PPO: {path_ppo}")

    # 2. DQN
    if os.path.exists(path_dqn):
        generate_dqn_plots([path_dqn], smoothing=smoothing_window)
    else:
        print(f"Brak pliku DQN: {path_dqn}")

    # 3. SAC-LSTM
    if os.path.exists(path_sac):
        generate_sac_plots([path_sac], smoothing=smoothing_window)
    else:
        print(f"Brak pliku SAC: {path_sac}")

    # 4. PORÓWNANIE (COMPARE)
    # Jeśli chcesz zestawić wszystkie na jednym wykresie
    print("\n=== Generowanie wykresów PORÓWNAWCZYCH (COMPARE) ===")
    all_files = []
    if os.path.exists(path_ppo): all_files.append(path_ppo)
    if os.path.exists(path_dqn): all_files.append(path_dqn)
    if os.path.exists(path_sac): all_files.append(path_sac)

    if len(all_files) > 1:
        prefix = "COMPARE_"
        # POPRAWIONE NAZWY ARGUMENTÓW PONIŻEJ (smooth_factor zamiast smoothing):
        plot_multi_algo_single_feature(all_files, "episode", "total_score", smooth_factor=smoothing_window, prefix=prefix)
        plot_multi_algo_single_feature(all_files, "episode", "avg_speed", smooth_factor=smoothing_window, prefix=prefix)
        plot_multi_algo_single_feature(all_files, "episode", "max_speed", smooth_factor=smoothing_window, prefix=prefix)
    else:
        print("Za mało plików do porównania (potrzebne min. 2).")