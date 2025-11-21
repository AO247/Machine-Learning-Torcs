import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import string


##Various algorithms log order per column
a2c_columns={"episode":0,"score":1,"total_loss":2,"policy_loss":3,"value_loss":4}
per_ddpg_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"critic_loss":6}
#dqn_columns = {"episode":0,"episode_step":1,"total_step":2,"total_score":3,"unused":4,"total_loss":5,"avg_q_value":6,"track_name":7,"race position":8,"max_speed":9,"avg_speed":10}
#dqn_columns = {"episode":0,"episode_step":1,"total_step":2,"total_score":3,"epsilon":4,"loss":5,"avg_q_value":6} ##This is not working with Torcs_dqn_5400
dqn_columns = {
    "episode": 0,
    "episode_step": 1,
    "total_step": 2,
    "total_score": 3,
    "epsilon": 4,
    "total_loss": 5,
    "avg_q_value": 6,
    "avg_time_cost": 7, # Dodajmy to dla spójności
    "track_name": 8,
    "race position": 9,
    "max_speed": 10,
    "avg_speed": 11
}

# ZMODYFIKOWANO: Usunięto 'max_rolling_total_score', aby pasowało do formatu logów z agent.py
sac_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"qf_1_loss":6,"qf_2_loss":7,"vf_loss":8,"alpha_loss":9,"track_name":10,"race_position":11,"max_speed":12,"avg_speed":13}
t3d_columns ={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor:loss":5,"critic1_loss":6,"critic2_loss":7,"track_name":8,"race_position":9,"max_speed":10,"avg_speed":11}

algo_column_list={
    "a2c":a2c_columns,
    "per-ddpg":per_ddpg_columns,
    "dqn": dqn_columns,
    "sac": sac_columns,
    "SAC": sac_columns,
    "SACLSTM": sac_columns,
    "sac-lstm": sac_columns, # DODANO: Obsługa formatu nazwy pliku 'sac-lstm'
    "t3d": t3d_columns
}



def get_column_indice(algo,column_name):
    columns = algo_column_list[algo]
    column_indice=columns[column_name]
    return column_indice

def get_column_values(algo,column_name):
    columns = algo_column_list[algo]
    column_indice=columns[column_name]
    return column_indice

def get_color(color_index):
    if color_index==0:
        return "red"
    elif color_index==1:
        return "blue"
    elif color_index==2:
        return "green"
    else:
        return "magenta"



def plot_multi_algo_single_feature(files, x_column, y_column, smooth_factor=100):
    color_index = 0

    # find min_episode for plots
    # load logs
    all_logs = []
    for logfilename_name in files:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_episode = np.min([len(logs) for logs in all_logs])

    for file in files:
        # Find algorithm type
        algo = file.split("_")[1]
        print("algo:" + algo)

        # Get column indices
        x_indice = get_column_indice(algo, x_column)
        y_indice = get_column_indice(algo, y_column)
        print(
            "x_column:" + x_column + " indice:" + str(x_indice) + " y_column:" + y_column + " indice:" + str(y_indice))

        # Get column values
        x_values, y_values = read_log_file(file, x_indice, y_indice)

        # Moving average y_values
        y_values = smoother(y_values, smooth_factor)

        #Cut exceeding episodes
        y_values = y_values[:min_episode]
        x_values = x_values[:len(y_values)] ##smoothing reduces the array length a bit


        # Get line color and advance to new color
        color = get_color(color_index)
        color_index += 1

        # Plot
        #plt.plot(y_values, color=color, label=algo)
        plt.plot(x_values,y_values, color=color, label=algo)

        plt.xlabel(x_column)
        plt.ylabel(y_column)
    plt.legend()
    plt.show()

def plot_algo_per_track(files, x_column, y_column,tracks, smooth_factor=100,graph_title=""):
    color_index = 0
    plt.figure()
    # find min_iteration for plots
    # load logs
    all_logs = []
    for logfilename_name in files:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_iteration = np.min([len(logs) for logs in all_logs])

    for track in tracks:
        for file in files:
            # Find algorithm type
            # POPRAWKA: Użyj os.path.basename
            basename = os.path.basename(file)
            algo = basename.split("_")[1]
            print("algo:" + algo)

            # Get column indices
            x_indice = get_column_indice(algo, x_column)
            y_indice = get_column_indice(algo, y_column)
            print(
                "x_column:" + x_column + " indice:" + str(x_indice) + " y_column:" + y_column + " indice:" + str(y_indice))

            # Get column values
            df = read_log_file_to_df(file)

            #df["track_name"].unique()

            # Only relevant tracks
            if len(tracks)>0:
                df = df[df["track_name"]==track]

            x_values = df[x_column]
            y_values = df[y_column]

            # Moving average y_values
            y_values = smoother(y_values, smooth_factor)

            #Cut exceeding episodes
            y_values = y_values[:min_iteration]
            x_values = x_values[:len(y_values)] ##smoothing reduces the array length a bit

            # Get line color and advance to new color
            color = get_color(color_index)
            color_index += 1

            # Plot
            #plt.plot(y_values, color=color, label=algo)
            #plt.plot(x_values,y_values, color=color, label=algo+"-"+track)
            plt.plot(x_values, y_values, label=algo + "-" + track)
            plt.savefig('graphs/test' + algo+".jpg", bbox_inches='tight')

            plt.xlabel(x_column)
            plt.ylabel(y_column)

    plt.legend()
    plt.title(graph_title)
    persist_figure(plt,"tracks")
    plt.show()


def smoother(array, ws):
    """ Return smoothed array by the mean filter """
    return np.array([sum(array[i:i+ws])/ws for i in range(len(array) - ws)])


def read_log_file(filename, x_indice, y_indice):
    x_values=[]
    y_values=[]

    file = open(filename, "r")

    #skip first 2 lines
    file.readline()
    file.readline()

    #read lines and get values
    for line in file:
        tokens = line.split(";")
        x_value=int(tokens[x_indice])
        y_value = float(tokens[y_indice])
        x_values.append(x_value)
        y_values.append(y_value)

    return x_values,y_values

def read_log_file_to_df(filename):
    # Find algorithm type
    # POPRAWKA: Użyj os.path.basename, aby wyodrębnić tylko nazwę pliku
    basename = os.path.basename(filename)
    algo = basename.split("_")[1]

    # Reszta funkcji pozostaje bez zmian
    file = open(filename, "r")
    columns=algo_column_list[algo].keys()
    column_names= list(columns)
    #read file to dataframe skipping the initial 2 lines
    df=pd.read_csv(file, names=column_names,sep = ';', skiprows=2)

    return df

def persist_figure(plt,title):
    rnd_filename= ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    plt.savefig('graphs/' + title+"_"+rnd_filename+".jpg", bbox_inches='tight')


def plot_same_algo_different_runs(logfilename_name_pairs, texts=[[""] * 3], smooth_factor=100):

    # find min_iteration for plots
    # load logs
    all_logs = []
    for logfilename_name,name in logfilename_name_pairs:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_iteration = np.min([len(logs) for logs in all_logs])


    for i, (title, xlabel, ylabel,y_axis_title) in enumerate(texts):

        #load logs
        all_logs=[]
        for logfilename_name,name in logfilename_name_pairs:
            logs = read_log_file_to_df(logfilename_name)
            all_logs.append(logs)

        #cut lenght to the min epiosode of logs
        min_episode = np.min([len(logs) for logs in all_logs])
        all_trimmed_logs =[]
        for logs in all_logs:
            all_trimmed_logs.append(logs.head(min_episode))

        smoothed_logs = np.stack([smoother(logs[ylabel], smooth_factor) for logs in all_trimmed_logs])

        std_logs = np.std(smoothed_logs, axis=0)
        mean_logs = np.mean(smoothed_logs, axis=0)
        max_logs = np.max(smoothed_logs, axis=0)
        min_logs = np.min(smoothed_logs, axis=0)

        plt.xlabel(xlabel)
        plt.ylabel(y_axis_title)
        plt.title(name)

        #Cut missing steps due to smoothing
        x_values = logs[xlabel][:len(mean_logs)]

        plt.plot(x_values,mean_logs, label=title)

        plt.legend()
        plt.fill_between(np.arange(len(mean_logs)),
                         np.minimum(mean_logs+std_logs, max_logs),
                         np.minimum(mean_logs-std_logs, min_logs),
                         alpha=0.4)

    persist_figure(plt,y_axis_title+"_"+name);
    plt.show()

def plot_multi_algo_single_feature(files, x_column, y_column, smooth_factor=100):
    color_index = 0
    plt.figure() # DODANO: Tworzy nową figurę dla każdego wykresu, aby się nie nakładały

    # find min_episode for plots
    # load logs
    all_logs = []
    for logfilename_name in files:
        logs = read_log_file_to_df(logfilename_name)
        all_logs.append(logs)

    # cut lenght to the min epiosode of logs
    min_episode = np.min([len(logs) for logs in all_logs])

    for file in files:
        # Find algorithm type
        # POPRAWKA: Poprawne wyodrębnianie nazwy algorytmu ze ścieżki
        basename = os.path.basename(file)
        algo = basename.split("_")[1]
        print("algo:" + algo)

        # Get column indices
        x_indice = get_column_indice(algo, x_column)
        y_indice = get_column_indice(algo, y_column)
        print(
            "x_column:" + x_column + " indice:" + str(x_indice) + " y_column:" + y_column + " indice:" + str(y_indice))

        # Get column values
        x_values, y_values = read_log_file(file, x_indice, y_indice)

        # Moving average y_values
        y_values = smoother(y_values, smooth_factor)

        #Cut exceeding episodes
        y_values = y_values[:min_episode]
        x_values = x_values[:len(y_values)] ##smoothing reduces the array length a bit


        # Get line color and advance to new color
        color = get_color(color_index)
        color_index += 1

        # Plot
        plt.plot(x_values,y_values, color=color, label=algo)

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{y_column} vs. {x_column}") # DODANO: Tytuł wykresu
    plt.legend()
    plt.grid(True) # DODANO: Siatka dla lepszej czytelności
    persist_figure(plt, f"{y_column}_vs_{x_column}") # Zapisz figurę
    plt.show()
##TESTS
##TESTS
if __name__ == "__main__":
    # Zmień na plik, który chcesz analizować
    log_filenames = ["logs/one_car/Torcs_sac-lstm_84389f4_fin.txt"]

    # --- Analiza dla 'total_score' ---
    all_scores = []
    for log_filename in log_filenames:
        print(f"--- Analizowanie pliku: {log_filename} ---")

        if not os.path.exists(log_filename):
            print(f"BŁĄD: Plik {log_filename} nie został znaleziony.")
            continue

        # POPRAWKA: Użyj os.path.basename, aby wyodrębnić tylko nazwę pliku
        try:
            basename = os.path.basename(log_filename) # Pobierz np. "Torcs_sac-lstm_84389f4_fin.txt"
            algo = basename.split('_')[1]             # Pobierz "sac-lstm"
            column_dict = algo_column_list[algo]
            score_column_index = column_dict["total_score"]
        except (IndexError, KeyError):
            print(f"BŁĄD: Nie można określić algorytmu lub znaleźć kolumny 'total_score' dla pliku: {log_filename}")
            continue

        # Pomiń pierwsze dwie linie nagłówka w pliku logów
        with open(log_filename, 'r') as file:
            lines = file.readlines()[2:] # Pomijamy pierwsze dwie linie
            for line in lines:
                parts = line.strip().split(';')
                try:
                    if len(parts) > score_column_index:
                        score = float(parts[score_column_index])
                        all_scores.append(score)
                except (ValueError, IndexError):
                    # Ignoruj linie, które nie są poprawnymi danymi (np. puste linie lub nagłówki)
                    pass

    if all_scores:
        scores_array = np.array(all_scores)
        max_score = np.max(scores_array)
        std_dev = np.std(scores_array)

        print("\n--- Wyniki analizy dla Total Score ---")
        print(f"Liczba przeanalizowanych epizodów: {len(all_scores)}")
        print(f"Maksymalny wynik (Max Score): {max_score:.2f}")
        print(f"Odchylenie standardowe (Standard Deviation): {std_dev:.2f}")
    else:
        print("\nNie znaleziono żadnych danych 'total_score' do analizy w podanych plikach.")

    # --- Analiza dla prędkości ---
    all_max_speeds = []
    all_avg_speeds = []
    for log_filename in log_filenames:
        if not os.path.exists(log_filename):
            continue

        # POPRAWKA: Użyj os.path.basename, aby wyodrębnić tylko nazwę pliku
        try:
            basename = os.path.basename(log_filename)
            algo = basename.split('_')[1]
            column_dict = algo_column_list[algo]
            max_speed_column_index = column_dict["max_speed"]
            avg_speed_column_index = column_dict["avg_speed"]
        except (IndexError, KeyError):
            print(f"BŁĄD: Nie można określić algorytmu lub znaleźć kolumn prędkości dla pliku: {log_filename}")
            continue

        # Pomiń pierwsze dwie linie nagłówka w pliku logów
        with open(log_filename, 'r') as file:
            lines = file.readlines()[2:] # Pomijamy pierwsze dwie linie
            for line in lines:
                parts = line.strip().split(';')
                try:
                    if len(parts) > max(max_speed_column_index, avg_speed_column_index):
                        max_speed = float(parts[max_speed_column_index])
                        avg_speed = float(parts[avg_speed_column_index])
                        all_max_speeds.append(max_speed)
                        all_avg_speeds.append(avg_speed)
                except (ValueError, IndexError):
                     # Ignoruj linie, które nie są poprawnymi danymi
                    pass

    if all_max_speeds:
        max_speeds_array = np.array(all_max_speeds)
        max_of_max_speed = np.max(max_speeds_array)
        std_dev_max_speed = np.std(max_speeds_array)

        print("\n--- Wyniki analizy dla Max Speed ---")
        print(f"Liczba przeanalizowanych epizodów: {len(all_max_speeds)}")
        print(f"Najwyższa maksymalna prędkość (Max of Max Speed): {max_of_max_speed:.2f}")
        print(f"Odchylenie standardowe (Std Dev of Max Speed): {std_dev_max_speed:.2f}")
    else:
        print("\nNie znaleziono danych dla 'max_speed'.")

    if all_avg_speeds:
        avg_speeds_array = np.array(all_avg_speeds)
        max_of_avg_speed = np.max(avg_speeds_array)
        std_dev_avg_speed = np.std(avg_speeds_array)

        print("\n--- Wyniki analizy dla Avg Speed ---")
        print(f"Liczba przeanalizowanych epizodów: {len(all_avg_speeds)}")
        print(f"Najwyższa średnia prędkość (Max of Avg Speed): {max_of_avg_speed:.2f}")
        print(f"Odchylenie standardowe (Std Dev of Avg Speed): {std_dev_avg_speed:.2f}")
    else:
        print("\nNie znaleziono danych dla 'avg_speed'.")

    log_file = ["logs/one_car/Torcs_sac-lstm_84389f4_fin.txt"]

    # Sprawdzenie, czy plik istnieje
    if not os.path.exists(log_file[0]):
        print(f"BŁĄD: Plik {log_file[0]} nie został znaleziony.")
    else:
        # Ustawiamy większy współczynnik wygładzania
        WYGLADZANIE = 200  # <-- ZDEFINIOWANA NOWA, WIĘKSZA WARTOŚĆ

        # 1. Wykres: oś y avg_speed, oś x episode
        print("\n--- Generowanie wykresu: Średnia prędkość od epizodu ---")
        plot_multi_algo_single_feature(
            files=log_file,
            x_column="episode",
            y_column="avg_speed",
            smooth_factor=WYGLADZANIE  # Użycie nowej wartości
        )

        # 2. Wykres: oś y max_speed, oś x episode
        print("\n--- Generowanie wykresu: Maksymalna prędkość od epizodu ---")
        plot_multi_algo_single_feature(
            files=log_file,
            x_column="episode",
            y_column="max_speed",
            smooth_factor=WYGLADZANIE  # Użycie nowej wartości
        )

        # 3. Wykres: oś y score, oś x episode
        print("\n--- Generowanie wykresu: Wynik od epizodu ---")
        plot_multi_algo_single_feature(
            files=log_file,
            x_column="episode",
            y_column="total_score",
            smooth_factor=WYGLADZANIE  # Użycie nowej wartości
        )