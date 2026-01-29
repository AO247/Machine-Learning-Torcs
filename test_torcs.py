import argparse
import importlib
import os
import time
import multiprocessing
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace
import numpy as np

# Importujemy środowiska
from env import torcs_envs as torcs

# --- KONFIGURACJA ---
PORTS = [3101, 3102, 3103]
XML_PATH = '/usr/local/share/games/torcs/config/raceman/quickrace.xml'
NUM_RACES = 20


def get_env_class(algo_name):
    if algo_name == "dqn":
        return torcs.DiscretizedEnv
    elif algo_name in ["sac", "sac-lstm", "ppo"]:
        return torcs.ContinuousEnv
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def set_quickrace_render_mode(render=True):
    try:
        tree = ET.parse(XML_PATH)
        root = tree.getroot()
        for section in root.findall('section'):
            if section.get('name') == "Quick Race":
                for attr in section.findall('attstr'):
                    if attr.get('name') == "display mode":
                        val = "normal" if render else "results only"
                        attr.set('val', val)
                        tree.write(XML_PATH)
                        return
    except Exception as e:
        print(f"[ERROR] Failed to modify quickrace.xml: {e}")


def agent_worker(rank, algo, model_path, reward_type, track, port, result_queue, sync_barrier):
    """
    Uruchamia agenta i wykonuje NUM_RACES wyścigów z synchronizacją.
    """
    # 1. Konfiguracja
    args = SimpleNamespace()
    args.algo = algo
    args.load_from = model_path
    args.test = True
    args.render = False
    args.log = False
    args.seed = 777 + rank
    args.reward_type = reward_type
    args.track = track
    args.max_episode_steps = 10000
    args.num_stack = 1

    state_filter = [1., 3., 10.]
    action_filter = [1., 3., 10.]

    agent_results = []

    # 2. Środowisko
    EnvClass = get_env_class(algo)
    env = None

    try:
        if algo == "dqn":
            env = EnvClass(port=port, nstack=1, reward_type=reward_type, track=track,
                           state_filter=state_filter, action_filter=None, action_count=21, client_mode=True)
        else:
            env = EnvClass(port=port, nstack=1, reward_type=reward_type, track=track,
                           state_filter=state_filter, action_filter=action_filter, client_mode=True)

        # 3. Agent
        module = importlib.import_module("torcs." + algo)
        agent = module.init(env, args)

        print(f"[{algo.upper()}] Connected on port {port}. Waiting for race start...")

        # 4. PĘTLA WYŚCIGÓW
        for race_idx in range(1, NUM_RACES + 1):

            # --- FIX: STAGGERED RESET (Szeregowanie połączeń) ---
            # Żeby nie zabić serwera 3 zapytaniami na raz, robimy odstępy.
            # Agent 0 czeka 0s, Agent 1 czeka 2s, Agent 2 czeka 4s.
            if race_idx > 1:
                sleep_time = rank * 2.0
                print(f"[{algo.upper()}] Waiting {sleep_time}s before reconnection...")
                time.sleep(sleep_time)

            # Resetujemy klienta
            obs = env.reset(relaunch=False, sampletrack=False)

            # Inicjalizacja LSTM (bezpieczna)
            hx, cx = None, None
            if hasattr(agent, 'actor') and hasattr(agent.actor, 'init_lstm_states'):
                hx, cx = agent.actor.init_lstm_states(1)

            done = False
            total_score = 0
            speeds = []

            # Jazda
            while not done:
                if algo == "sac-lstm":
                    if hx is not None:
                        action, hx, cx = agent.select_action(obs, hx, cx)
                    else:
                        action = agent.select_action(obs) if hasattr(agent, 'select_action') else np.zeros(
                            env.action_dim)
                else:
                    action = agent.select_action(obs)

                obs, reward, done, info = env.step(action)
                total_score += reward

                spd = env.last_speed if hasattr(env, 'last_speed') else 0.0
                speeds.append(spd)

            # Statystyki lokalne
            # Ignorujemy info['place'], bo TORCS kłamie. Policzmy to na końcu.
            max_spd = max(speeds) if speeds else 0.0
            avg_spd = sum(speeds) / len(speeds) if speeds else 0.0

            print(f"[{algo.upper()}] Race {race_idx} Finished. Score: {total_score:.2f}. Waiting for others...")

            agent_results.append({
                "algo": algo,
                "rank": rank,
                "race_idx": race_idx,
                "score": total_score,
                "max_speed": max_spd,
                "avg_speed": avg_spd
            })

            # --- BARIERA SYNCHRONIZACYJNA ---
            try:
                sync_barrier.wait(timeout=180)  # Czekamy aż wszyscy skończą
            except multiprocessing.BrokenBarrierError:
                print(f"[{algo.upper()}] Barrier broken! Exiting...")
                break

            print(f"[{algo.upper()}] Sync complete.")

        result_queue.put(agent_results)

    except Exception as e:
        print(f"[{algo.upper()}] CRASHED: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put([])
    finally:
        if env:
            try:
                env.close()
            except:
                pass


def launch_torcs_server(render_mode):
    set_quickrace_render_mode(render=render_mode)
    print("[SERVER] Launching TORCS...")
    os.system('pkill torcs')
    time.sleep(0.5)
    os.system('torcs -nofuel -nodamage -nolaptime -p 3101 &')
    time.sleep(1.0)
    os.system('sh autostart.sh')
    time.sleep(4.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    cmd_args = parser.parse_args()

    render_enabled = not cmd_args.no_render

    models = [
        {"algo": "sac-lstm", "path": "save/Torcs_sac-lstm_f312ffd_ep_3200.pt"},
        {"algo": "ppo", "path": "save/Torcs_ppo_f312ffd_ep_5400.pt"},
        {"algo": "dqn", "path": "save/Torcs_dqn_f312ffd_ep_3800.pt"},
    ]

    reward_type = "extra_github"
    track_name = "e-track-1"

    try:
        launch_torcs_server(render_enabled)

        result_queue = multiprocessing.Queue()
        sync_barrier = multiprocessing.Barrier(len(models))

        processes = []
        for i, model_cfg in enumerate(models):
            port = PORTS[i]
            p = multiprocessing.Process(
                target=agent_worker,
                args=(
                i, model_cfg["algo"], model_cfg["path"], reward_type, track_name, port, result_queue, sync_barrier)
            )
            processes.append(p)
            p.start()
            # Tutaj też odstęp na start
            time.sleep(2.0)

        all_results = []
        print("\n[MAIN] Waiting for results...")

        for _ in range(len(models)):
            agent_data = result_queue.get()
            all_results.extend(agent_data)

        for p in processes:
            p.join()

        # --- FIX RANKINGU (Liczymy miejsca w Pythonie) ---
        # Sortujemy po numerze wyścigu
        all_results.sort(key=lambda x: x['race_idx'])

        # Grupujemy wyniki per wyścig i przydzielamy miejsca
        final_results = []
        current_race_idx = -1
        race_batch = []

        # Prosta pętla do grupowania i sortowania wewnątrz grupy
        for res in all_results:
            if res['race_idx'] != current_race_idx:
                # Przetwórz poprzednią grupę
                if race_batch:
                    # Sortuj malejąco po Score (kto ma więcej, ten wygrał)
                    race_batch.sort(key=lambda x: x['score'], reverse=True)
                    # Przydziel miejsca
                    for rank, r in enumerate(race_batch):
                        r['place'] = rank + 1
                    final_results.extend(race_batch)

                current_race_idx = res['race_idx']
                race_batch = []

            race_batch.append(res)

        # Ostatnia grupa
        if race_batch:
            race_batch.sort(key=lambda x: x['score'], reverse=True)
            for rank, r in enumerate(race_batch):
                r['place'] = rank + 1
            final_results.extend(race_batch)

        # Zapis
        algo_names = [m['algo'] for m in models]
        filename = f"test_{'_'.join(algo_names)}.txt"

        print(f"\n[MAIN] Saving results to {filename}...")

        with open(filename, "w") as f:
            f.write("Race;Algo;Place;Score;MaxSpeed;AvgSpeed\n")
            for res in final_results:
                line = f"{res['race_idx']};{res['algo']};{res['place']};{res['score']:.2f};{res['max_speed']:.2f};{res['avg_speed']:.2f}\n"
                f.write(line)
                print(line.strip())

        print("[MAIN] SUCCESS.")

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted.")
    except Exception as e:
        print(f"\n[MAIN] Error: {e}")
    finally:
        print("[MAIN] Closing TORCS...")
        os.system('pkill torcs')
        sys.exit(0)