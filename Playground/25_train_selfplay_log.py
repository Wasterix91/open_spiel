import os
import re
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import ppo_local_2 as ppo
from tqdm import trange
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict

# === 1️⃣ Spielinitialisierung ===
game = pyspiel.load_game("president", {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
})
env = rl_environment.Environment(game)

# === 2️⃣ PPO-Agenten (Self-Play) ===
agents = [
    ppo.PPOAgent(
        env.observation_spec()["info_state"][0],
        env.action_spec()["num_actions"],
        ppo.DEFAULT_CONFIG
    )
    for _ in range(4)
]

# === 3️⃣ Versionierung & Ordnerstruktur ===
base_dir = os.path.dirname(__file__)
models_root = os.path.join(base_dir, "models")
prefix = "selfplay_president_"

existing = [
    name for name in os.listdir(models_root)
    if os.path.isdir(os.path.join(models_root, name)) and re.match(rf"{prefix}\d+", name)
]
numbers = [int(re.findall(r"\d+", name)[0]) for name in existing] if existing else [0]
next_num = max(numbers) + 1
version_name = f"{prefix}{next_num:02d}"

train_dir = os.path.join(models_root, version_name, "train")
test_dir = os.path.join(models_root, version_name, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

print(f"✅ Self-Play-Training gestartet. Version: {version_name}")

# === 🔁 Eval-Funktion gegen Random-Gegner ===
def evaluate_against_random(env, agent, num_games=100):
    wins = 0
    for _ in range(num_games):
        time_step = env.reset()
        while not time_step.last():
            p = time_step.observations["current_player"]
            legal = time_step.observations["legal_actions"][p]

            if p == 0:
                obs = time_step.observations["info_state"][p]
                agent_out = agent.step(time_step, legal)
                action = agent_out.action if agent_out else np.random.choice(legal)
            else:
                action = np.random.choice(legal)

            time_step = env.step([action])

        winner = np.argmax(time_step.rewards)
        if winner == 0:
            wins += 1
    return wins / num_games

# === 4️⃣ Training ===
num_episodes = 20_000
returns = [[] for _ in range(4)]
win_rates = []
eval_winrates = []  # (episode, winrate)
start_counts = defaultdict(int)
progress = trange(1, num_episodes + 1, desc="Training", unit="ep")

for ep in progress:
    time_step = env.reset()
    start_player = time_step.observations["current_player"]
    start_counts[start_player] += 1
    steps = 0

    while not time_step.last():
        p = time_step.observations["current_player"]
        obs = time_step.observations["info_state"][p]
        legal = time_step.observations["legal_actions"][p]

        agent_out = agents[p].step(time_step, legal)
        action = agent_out.action if agent_out else np.random.choice(legal)
        time_step = env.step([action])
        steps += 1

        if steps > 200:
            progress.write(f"⚠️ Episode {ep} abgebrochen bei 200 Schritten (Sicherheitsabbruch)")
            break

    for pid in range(4):
        agents[pid].step(time_step, [0])
        returns[pid].append(time_step.rewards[pid])

    winner = np.argmax(time_step.rewards)
    win_rates.append(1 if winner == 0 else 0)

    if ep % 100 == 0:
        ret_avgs = [np.mean(r[-100:]) for r in returns]
        win_avg = np.mean(win_rates[-100:]) * 100
        msg = " | ".join([f"P{pid}: {ret:.2f}" for pid, ret in enumerate(ret_avgs)])
        progress.write(f"[Ep {ep}] Ø Return (100): {msg} | Winrate P0: {win_avg:.1f}%")

    if ep % 500 == 0:
        eval_wr = evaluate_against_random(env, agents[0])
        eval_winrates.append((ep, eval_wr))
        progress.write(f"[Eval] Ep {ep}: P0 vs Random Winrate: {eval_wr*100:.1f}%")

# === 5️⃣ Speichern ===
for pid, agent in enumerate(agents):
    agent.save(os.path.join(train_dir, f"{version_name}_agent_p{pid}"))
    np.save(os.path.join(train_dir, f"returns_p{pid}.npy"), returns[pid])

np.save(os.path.join(train_dir, "win_rates.npy"), win_rates)

for f in os.listdir(train_dir):
    shutil.copy(os.path.join(train_dir, f), os.path.join(test_dir, f))

print(f"✅ Modelle & Daten gespeichert in: {train_dir} und {test_dir}")

# === 6️⃣ Winrate-Plot für Spieler 0 ===
window = 100
smoothed = np.convolve(win_rates, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12,6))
plt.plot(win_rates, color='lightblue', label="Win (0/1)")
plt.plot(range(window - 1, len(win_rates)), smoothed, color='orange', label=f"Moving Avg ({window})")
plt.title(f"{version_name}: Winrate P0 Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Win (P0)")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)

plot_path = os.path.join(train_dir, "ppo_winrate_over_time.png")
plt.savefig(plot_path)
print(f"📊 Winrate-Plot gespeichert: {plot_path}")
plt.show()

# === 7️⃣ Eval-Plot gegen Random Agents ===
if eval_winrates:
    eval_eps, eval_scores = zip(*eval_winrates)
    np.save(os.path.join(train_dir, "eval_vs_random.npy"), eval_winrates)

    plt.figure(figsize=(12,6))
    plt.plot(eval_eps, eval_scores, marker='o')
    plt.title(f"{version_name}: Evaluation vs. 3 Random Agents (P0)")
    plt.xlabel("Training Episode")
    plt.ylabel("Winrate P0")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.savefig(os.path.join(train_dir, "eval_vs_random_plot.png"))
    print("📊 Evaluation gegen Random Agents gespeichert & geplottet.")
    plt.show()

# === 8️⃣ Startspieler-Statistik ===
print("\n=== Startspieler-Statistik ===")
total_starts = sum(start_counts.values())
for pid in range(4):
    count = start_counts[pid]
    share = 100 * count / total_starts if total_starts else 0
    print(f"Player {pid} started {count} times ({share:.2f}%)")
