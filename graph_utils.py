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
    "avg_speed": 11,
    "time": 12
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
    "avg_speed": 13,
    "time": 14
}
ppo_columns = {
    "episode": 0,
    "episode_step": 1,
    "total_step": 2,
    "total_score": 3,
    "total_loss": 4,
    "actor_loss": 5,
    "approx_kl": 6,
    "clip_frac": 7,
    "vf_loss": 8,
    "entropy": 9,
    "track_name": 10,
    "race_position": 11,
    "max_speed": 12,
    "avg_speed": 13,
    "time": 14
}

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
    "ppo": ppo_columns,
    "PPO": ppo_columns
}


def get_column_indice(algo, column_name):
    if algo == "ppo":
        if column_name == "alpha_loss": return ppo_columns["entropy"]  # Mapowanie PPO Entropy na Alpha

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

            x_value = float(tokens[x_indice])
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
    filename = f"graphs/{title}_{rnd_filename}.jpg"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved graph: {filename}")


def plot_multi_algo_single_feature(files, x_column, y_column, smooth_factor=100, prefix=""):
    plt.figure(figsize=(10, 6))

    has_data = False

    for file in files:
        basename = os.path.basename(file)
        try:
            # Zakładamy format nazwy: Torcs_ALGO_hash.txt
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
            # ZMIANA: Przekazujemy filename, żeby rozróżnić serie
            color = get_algo_color(algo, filename=file)

            # Formatowanie etykiety: Dodajemy nazwę folderu/wariantu
            variant = "OneCar" if "one_car" in file else "SevenCars"
            label_text = f"{algo.upper()} ({variant})"

            plt.plot(x_vals, y_vals, color=color, label=label_text, linewidth=1.5)
            has_data = True

    if not has_data:
        print(f"No data found for {y_column} vs {x_column}")
        plt.close()
        return

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    # plt.title(f"{prefix}{y_column} vs {x_column} (Smooth: {smooth_factor})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    persist_figure(plt, f"{prefix}{y_column}_vs_{x_column}")
    plt.close()



def generate_ppo_plots(log_files, smoothing=100):
    print("\n=== Generowanie wykresów dla PPO ===")
    prefix = "PPO_"
    plot_multi_algo_single_feature(log_files, "episode", "total_score", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "avg_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "max_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "alpha_loss", smoothing, prefix)  # Entropy


def generate_dqn_plots(log_files, smoothing=100):
    print("\n=== Generowanie wykresów dla DQN ===")
    prefix = "DQN_"
    plot_multi_algo_single_feature(log_files, "episode", "total_score", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "avg_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "max_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "total_loss", smoothing, prefix)


def generate_sac_plots(log_files, smoothing=100):
    print("\n=== Generowanie wykresów dla SAC-LSTM ===")
    prefix = "SAC-LSTM_"
    plot_multi_algo_single_feature(log_files, "episode", "total_score", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "avg_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "max_speed", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "alpha_loss", smoothing, prefix)
    plot_multi_algo_single_feature(log_files, "episode", "total_loss", smoothing, prefix)


def get_algo_color(algo_name, filename=""):
    """
    Przypisuje kolor na podstawie algorytmu i nazwy pliku.
    Jeśli to ten sam algorytm, ale inny plik (np. dqn2), daje inny kolor.
    """
    name = algo_name.lower()

    # Sprawdzenie konkretnych plików (Hardcoded dla Twojego przypadku)
    if "one_car" in filename:  # Wykrywamy, że to plik z folderu 'one_car'
        return "red"  # Druga seria będzie zawsze czerwona

    # Standardowe kolory dla pierwszej serii ('seven_cars')
    if "ppo" in name:
        return "green"
    elif "dqn" in name:
        return "blue"
    elif "sac" in name:
        return "orange"
    else:
        return "black"
# --- MAIN ---

LOG_FILE = "logs/seven_cars/Torcs_sac-lstm_f312ffd.txt"
OUTPUT_FILE = "sac_track_stats.csv"

# Indeksy kolumn dla SAC-LSTM (zgodnie z Twoimi logami)
# 0:Episode; 1:Step; 2:TotalStep; 3:Score; ... 10:Track; 11:Pos; 12:MaxSpd; 13:AvgSpd; 14:Time
IDX_SCORE = 3
IDX_TRACK = 10
IDX_MAX_SPD = 12
IDX_AVG_SPD = 13


def generate_excel_table():
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] Nie znaleziono pliku: {LOG_FILE}")
        return

    # Słownik do przechowywania danych: { 'nazwa_toru': { 'scores': [], 'max_speeds': [], 'avg_speeds': [] } }
    tracks_data = {}

    print(f"Przetwarzanie pliku: {LOG_FILE}...")

    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Pomijamy nagłówki i puste linie
            if not line or line.startswith(("Namespace", "{", "---", "[INFO]", "Ratio")):
                continue

            parts = line.split(";")

            # Sprawdzamy czy linia ma wystarczająco dużo danych
            if len(parts) <= 13:
                continue

            try:
                # Pobieramy dane
                track_name = parts[IDX_TRACK].strip()
                score = float(parts[IDX_SCORE])
                max_spd = float(parts[IDX_MAX_SPD])
                avg_spd = float(parts[IDX_AVG_SPD])

                # Inicjalizacja listy dla nowego toru
                if track_name not in tracks_data:
                    tracks_data[track_name] = {
                        'scores': [],
                        'max_speeds': [],
                        'avg_speeds': []
                    }

                # Dodawanie danych
                tracks_data[track_name]['scores'].append(score)
                tracks_data[track_name]['max_speeds'].append(max_spd)
                tracks_data[track_name]['avg_speeds'].append(avg_spd)

            except ValueError:
                continue

        # --- GENEROWANIE TABELI ---
        output_lines = []
        # Nagłówek dla Excela (średnik jako separator jest bezpieczny dla polskich ustawień Excela)
        output_lines.append("Track;Max Score;Avg Score;Max Speed;Avg Speed")

        # Sortujemy tory alfabetycznie
        sorted_tracks = sorted(tracks_data.keys())

        print("\n--- PODGLĄD WYNIKÓW (SAC-LSTM) ---")
        print(f"{'Track':<15} | {'Max Score':<10} | {'Avg Score':<10} | {'Max Speed':<10} | {'Avg Speed':<10}")
        print("-" * 70)

        for track in sorted_tracks:
            data = tracks_data[track]

            # Obliczenia
            max_score = np.max(data['scores'])
            avg_score = np.mean(data['scores'])

            # Dla Max Speed bierzemy maksymalną osiągniętą prędkość na danym torze (peak)
            max_speed_val = np.max(data['max_speeds'])

            # Dla Avg Speed bierzemy średnią ze średnich prędkości (tempo wyścigowe)
            avg_speed_val = np.mean(data['avg_speeds'])

            # Formatowanie do pliku (zminiamy kropkę na przecinek, jeśli Twój Excel tego wymaga,
            # ale standardowo w CSV programistycznym używa się kropki. Tu zostawiam kropkę).
            line_str = f"{track};{max_score:.2f};{avg_score:.2f};{max_speed_val:.2f};{avg_speed_val:.2f}"
            output_lines.append(line_str)

            # Podgląd w konsoli
            print(
                f"{track:<15} | {max_score:<10.2f} | {avg_score:<10.2f} | {max_speed_val:<10.2f} | {avg_speed_val:<10.2f}")

        # Zapis do pliku
        with open(OUTPUT_FILE, "w") as f:
            f.writelines([line + "\n" for line in output_lines])

        print("-" * 70)
        print(f"Gotowe! Plik '{OUTPUT_FILE}' został utworzony.")
        print("Możesz go otworzyć w Excelu (Dane -> Z tekstu/CSV -> Separator: średnik).")

    except Exception as e:
        print(f"[ERROR] Coś poszło nie tak: {e}")


if __name__ == "__main__":
    # generate_excel_table()

    path_ppo = "logs/seven_cars/Torcs_ppo_f312ffd.txt"
    path_ppo2 = "logs/one_car/Torcs_ppo_5542983.txt"
    path_dqn = "logs/seven_cars/Torcs_dqn_f312ffd.txt"
    path_dqn2 = "logs/one_car/Torcs_dqn_4b15edf.txt"
    path_sac = "logs/seven_cars/Torcs_sac-lstm_f312ffd.txt"
    #
    smoothing_window = 200

    # # 1. PPO
    # if os.path.exists(path_ppo):
    #     generate_ppo_plots([path_ppo], smoothing=smoothing_window)
    # else:
    #     print(f"Brak pliku PPO: {path_ppo}")
    #
    # # 2. DQN
    # if os.path.exists(path_dqn):
    #     generate_dqn_plots([path_dqn], smoothing=smoothing_window)
    # else:
    #     print(f"Brak pliku DQN: {path_dqn}")
    #
    # # # 3. SAC-LSTM
    # if os.path.exists(path_sac):
    #     generate_sac_plots([path_sac], smoothing=smoothing_window)
    # else:
    #     print(f"Brak pliku SAC: {path_sac}")
    #
    # # # 4. PORÓWNANIE
    #
    # smoothing_window = 400
    #
    # all_files = []
    # if os.path.exists(path_ppo): all_files.append(path_ppo)
    # if os.path.exists(path_dqn): all_files.append(path_dqn)
    # if os.path.exists(path_sac): all_files.append(path_sac)
    #
    # if len(all_files) > 1:
    #     prefix = "COMPARE_"
    #     plot_multi_algo_single_feature(all_files, "episode", "total_score", smooth_factor=smoothing_window, prefix=prefix)
    #     plot_multi_algo_single_feature(all_files, "episode", "avg_speed", smooth_factor=smoothing_window, prefix=prefix)
    #     plot_multi_algo_single_feature(all_files, "episode", "max_speed", smooth_factor=smoothing_window, prefix=prefix)
    # else:
    #     print("Za mało plików do porównania (potrzebne min. 2).")
    #
    # if len(all_files) > 1:
    #     prefix = "COMPARE_"
    #     plot_multi_algo_single_feature(all_files, "time", "total_score", smooth_factor=smoothing_window,
    #                                    prefix=prefix)
    #     plot_multi_algo_single_feature(all_files, "time", "avg_speed", smooth_factor=smoothing_window, prefix=prefix)
    #     plot_multi_algo_single_feature(all_files, "time", "max_speed", smooth_factor=smoothing_window, prefix=prefix)
    # else:
    #     print("Za mało plików do porównania (potrzebne min. 2).")
    # all_files2 = []
    # if os.path.exists(path_dqn): all_files2.append(path_dqn)
    # if os.path.exists(path_sac): all_files2.append(path_sac)
    # if len(all_files2) > 1:
    #     prefix = "COMPARE_"
    #     plot_multi_algo_single_feature(all_files2, "episode", "total_score", smooth_factor=smoothing_window, prefix=prefix)
    # else:
    #     print("Za mało plików do porównania (potrzebne min. 2).")

    all_files_dqn = []
    if os.path.exists(path_dqn): all_files_dqn.append(path_dqn)
    if os.path.exists(path_dqn2): all_files_dqn.append(path_dqn2)

    if len(all_files_dqn) > 1:
        prefix = "COMPARE_"
        plot_multi_algo_single_feature(all_files_dqn, "episode", "total_score", smooth_factor=smoothing_window, prefix=prefix)
        plot_multi_algo_single_feature(all_files_dqn, "episode", "avg_speed", smooth_factor=smoothing_window, prefix=prefix)
        plot_multi_algo_single_feature(all_files_dqn, "episode", "max_speed", smooth_factor=smoothing_window, prefix=prefix)
    else:
        print("Za mało plików do porównania (potrzebne min. 2).")

    all_files_ppo = []
    if os.path.exists(path_ppo): all_files_ppo.append(path_ppo)
    if os.path.exists(path_ppo2): all_files_ppo.append(path_ppo2)

    if len(all_files_ppo) > 1:
        prefix = "COMPARE_"
        plot_multi_algo_single_feature(all_files_ppo, "episode", "total_score", smooth_factor=smoothing_window,
                                       prefix=prefix)
        plot_multi_algo_single_feature(all_files_ppo, "episode", "avg_speed", smooth_factor=smoothing_window,
                                       prefix=prefix)
        plot_multi_algo_single_feature(all_files_ppo, "episode", "max_speed", smooth_factor=smoothing_window,
                                       prefix=prefix)
    else:
        print("Za mało plików do porównania (potrzebne min. 2).")