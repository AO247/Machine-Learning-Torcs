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

sac_columns={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor_loss":5,"qf_1_loss":6,"qf_2_loss":7,"vf_loss":8,"alpha_loss":9,"track_name":10,"race_position":11,"max_speed":12,"avg_speed":13,"max_rolling_total_score":14}
t3d_columns ={"episode":0,"episode_step":1,"total_step":2,"total_score":3,"total_loss":4,"actor:loss":5,"critic1_loss":6,"critic2_loss":7,"track_name":8,"race_position":9,"max_speed":10,"avg_speed":11}

algo_column_list={
    "a2c":a2c_columns,
    "per-ddpg":per_ddpg_columns,
    "dqn": dqn_columns,
    "sac": sac_columns,
    "SAC": sac_columns,
    "SACLSTM": sac_columns,
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
            algo = file.split("_")[1]
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
    algo = filename.split("_")[1]

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


##TESTS
if __name__ == "__main__":
    log_filenames = ["logs/Torcs_dqn_4b15edf.txt"]

    # # Wywołanie funkcji do rysowania wykresu
    # plot_multi_algo_single_feature(
    #     log_filenames,
    #     x_column="episode",
    #     y_column="max_speed",
    #     smooth_factor=250  # <-- KLUCZOWY PARAMETR Z ARTYKUŁU!
    # )


    # ##PLOT 1 Loss graphs min,max,mean,std of the runs with smoothing
    # ## with multiple features ( i.e. Max Speed , Average speed)
    # plot_texts = [
    #     [
    #         "total_step",#x_column_legend
    #         "total_step",#x_column value
    #         "total_loss",## y_column value
    #         "Loss",##y_axis title
    #     ],
    # ]
    # #plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM")], texts=plot_texts,smooth_factor=20)
    # #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN")],texts=plot_texts, smooth_factor=20)
    # plot_same_algo_different_runs([("logs/Torcs_dqn_4b15edf.txt", "DQN")],texts=plot_texts, smooth_factor=20)
    #
    #
    #
    #
    # ##PLOT 2 Multiple runs of same algorithm plots min,max,mean,std of the runs with smoothing
    # ## with multiple features ( i.e. Max Speed , Average speed)
    # plot_texts = [
    #     [
    #         "Max Speed",#x_column_legend
    #         "total_step", #x_column value
    #         "max_speed", ## y_column value
    #         "speed" ##y_axis title
    #     ],
    #     [
    #         "Average Speed",#x_column_legend
    #         "total_step",#x_column value
    #         "avg_speed",## y_column value
    #         "speed",##y_axis title
    #
    #     ]
    # ]
    #
    # #plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM"), ],texts=plot_texts, smooth_factor=50)
    # #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN"), ],texts=plot_texts, smooth_factor=50)
    # plot_same_algo_different_runs([("logs/Torcs_dqn_4b15edf.txt", "DQN")],texts=plot_texts, smooth_factor=50)
    #
    #
    # ##PLOT 3 - Rewards
    # ## with multiple features ( i.e. Max Speed , Average speed)
    # plot_texts = [
    #     [
    #         "episode_step",#x_column_legend
    #         "total_step", #x_column value
    #         "total_score", ## y_column value
    #         "Reward" ##y_axis title
    #     ]
    # ]
    #
    # #plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM")],texts=plot_texts, smooth_factor=100)
    # #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN") ],texts=plot_texts, smooth_factor=200)
    # plot_same_algo_different_runs([("logs/Torcs_dqn_4b15edf.txt", "DQN")],texts=plot_texts, smooth_factor=200)
    #
    # # ##PLOT 3.5 - Max Cumulative Rewards
    # # ## with multiple features ( i.e. Max Speed , Average speed)
    # # plot_texts = [
    # #     [
    # #         "episode_step",#x_column_legend
    # #         "total_step", #x_column value
    # #         "max_rolling_total_score", ## y_column value
    # #         "Max_Rolling_Reward" ##y_axis title
    # #     ]
    # # ]
    # #
    # # #plot_same_algo_different_runs([("releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log", "SACLSTM")],texts=plot_texts, smooth_factor=100)
    # # #plot_same_algo_different_runs([("releases/Torcs_dqn_a11ccc2.log", "DQN") ],texts=plot_texts, smooth_factor=200)
    # # plot_same_algo_different_runs([("logs/Torcs_dqn_4b15edf.txt", "DQN")],texts=plot_texts, smooth_factor=200)
    # #
    #
    #
    #
    # ##PLOT 4 Compare different algos against same feature (i.e. max_reward)
    # #x_column="episode"
    # x_column = "total_step"
    # y_column="total_score"
    # #log_filenames=["releases/TORCS_SACLSTM_512256128_L1_EP4000_N1_G99_trimmed.log", "releases/Torcs_sac_f622afd.log","releases/Torcs_dqn_a11ccc2.log"]
    # log_filenames = ["logs/Torcs_dqn_4b15edf.txt", "logs/Torcs_dqn_4b15edf.txt"]
    #
    # #plot_multi_algo_single_feature(log_filenames, x_column=x_column, y_column=y_column, smooth_factor=1000)
    #
    #
    # ##PLOT 5 Compare algos against tracks
    # x_column = "total_step"
    # y_column="max_speed"
    # #log_filenames=["releases/TORCS_SAC_N4_G99_2000.log", "releases/TORCS_SACLSTM_512256128_L1_EP2000_N1_G99.log"]
    # #log_filenames = ["releases/TORCS_SAC_N4_G99_2000_trimmed.log", "releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log"]
    # log_filenames = ["logs/Torcs_dqn_4b15edf.txt", "logs/Torcs_dqn_4b15edf.txt"]
    #
    # tracks_all=['e-track-2','alpine-1','g-track-1']
    # plot_algo_per_track(log_filenames, tracks=tracks_all, x_column=x_column, y_column=y_column, smooth_factor=10,graph_title="Max Speed per track and algo")
    #
    #
    # ##PLOT 6 Compare algos against tracks
    # x_column = "total_step"
    # y_column="total_score"
    # #log_filenames=["releases/TORCS_SAC_N4_G99_2000.log", "releases/TORCS_SACLSTM_512256128_L1_EP2000_N1_G99.log"]
    # #log_filenames = ["releases/TORCS_SAC_N4_G99_2000_trimmed.log", "releases/TORCS_SACLSTM_512256128_L1_EP3000_N1_G99_trimmed.log"]
    # log_filenames = ["logs/Torcs_dqn_4b15edf.txt", "logs/Torcs_dqn_4b15edf.txt"]
    # tracks_all = ['e-track-2', 'alpine-1', 'g-track-1']
    # plot_algo_per_track(log_filenames, tracks=tracks_all, x_column=x_column, y_column=y_column, smooth_factor=20,graph_title="Reward per track and algo")
    #
    #
    # Lista do przechowywania wszystkich wyników z plików
    all_scores = []

    # Pobieramy indeks kolumny, która nas interesuje
    score_column_index = dqn_columns["total_score"]

    for log_filename in log_filenames:
        print(f"--- Analizowanie pliku: {log_filename} ---")

        # Sprawdzamy, czy plik istnieje
        if not os.path.exists(log_filename):
            print(f"BŁĄD: Plik {log_filename} nie został znaleziony.")
            continue  # Przejdź do następnego pliku

        # Otwieramy plik i czytamy go linijka po linijce
        with open(log_filename, 'r') as file:
            for line in file:
                # Dzielimy linijkę po średniku, aby uzyskać poszczególne wartości
                parts = line.strip().split(';')

                # Używamy bloku try-except, aby zignorować nagłówki i inne nieprawidłowe linie
                try:
                    # Sprawdzamy, czy linia ma wystarczająco dużo kolumn
                    if len(parts) > score_column_index:
                        # Pobieramy wartość z kolumny "total_score" i konwertujemy ją na liczbę
                        score = float(parts[score_column_index])
                        all_scores.append(score)
                except ValueError:
                    # Ignorujemy linie, w których nie da się przekonwertować wartości na liczbę
                    # (np. nagłówki "episode;episode_step;...")
                    pass

    # Po wczytaniu wszystkich danych, sprawdzamy, czy mamy co analizować
    if all_scores:
        # Konwertujemy listę na tablicę NumPy dla łatwych i szybkich obliczeń
        scores_array = np.array(all_scores)

        # Obliczamy maksymalny wynik
        max_score = np.max(scores_array)

        # Obliczamy odchylenie standardowe
        std_dev = np.std(scores_array)

        print("\n--- Wyniki analizy ---")
        print(f"Liczba przeanalizowanych epizodów: {len(all_scores)}")
        print(f"Maksymalny wynik (Max Score): {max_score:.2f}")
        print(f"Odchylenie standardowe (Standard Deviation): {std_dev:.2f}")

    else:
        print("\nNie znaleziono żadnych danych do analizy w podanych plikach.")
    all_max_speeds = []
    all_avg_speeds = []

    # Pobieramy indeksy kolumn, które nas interesują
    max_speed_column_index = dqn_columns["max_speed"]
    avg_speed_column_index = dqn_columns["avg_speed"]

    for log_filename in log_filenames:
        print(f"--- Analizowanie pliku: {log_filename} ---")

        if not os.path.exists(log_filename):
            print(f"BŁĄD: Plik {log_filename} nie został znaleziony.")
            continue

        with open(log_filename, 'r') as file:
            for line in file:
                parts = line.strip().split(';')

                try:
                    # Sprawdzamy, czy linia ma wystarczająco dużo kolumn
                    if len(parts) > max(max_speed_column_index, avg_speed_column_index):
                        # Pobieramy wartości z odpowiednich kolumn i konwertujemy je na liczby
                        max_speed = float(parts[max_speed_column_index])
                        avg_speed = float(parts[avg_speed_column_index])

                        all_max_speeds.append(max_speed)
                        all_avg_speeds.append(avg_speed)
                except ValueError:
                    # Ignorujemy linie, których nie da się przekonwertować (nagłówki itp.)
                    pass

    # Analiza dla `max_speed`
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

    # Analiza dla `avg_speed`
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





