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

# === 1️⃣ Konfiguration ===
PLAYER_TYPES = ["ppo", "random", "random", "random"]
NUM_EPISODES = 30_000
GENERATE_PLOTS = False

# === ⚙️ Reward Shaping ===
REWARD_SHAPING = True
REWARD_MAPPING = [10, 5, 0, 0]  # Platz 1 → 10 Punkte, Platz 2 → 3 Punkte, Rest 0

# === 2️⃣ Spielinitialisierung ===
game = pyspiel.load_game("president", {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
})
env = rl_environment.Environment(game)

# === 3️⃣ Strategien definieren ===
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_combo_size(text):
    if "Single" in text: return 1
    if "Pair" in text: return 2
    if "Triple" in text: return 3
    if "Quad" in text: return 4
    return 1

def parse_rank(text):
    try:
        return RANK_TO_NUM[text.split()[-1]]
    except KeyError:
        return -1

def decode_actions(state):
    player = state.current_player()
    actions = state.legal_actions()
    return [(a, state.action_to_string(player, a)) for a in actions if a != 0]

def max_combo_strategy(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_combo_size(x[1]))[0] if decoded else 0

def aggressive_strategy(state):
    decoded = decode_actions(state)
    return max(decoded, key=lambda x: parse_rank(x[1]))[0] if decoded else 0

def single_only_strategy(state):
    decoded = decode_actions(state)
    singles = [x for x in decoded if "Single" in x[1]]
    return min(singles, key=lambda x: x[0])[0] if singles else 0

def random_action_strategy(state):
    return np.random.choice(state.legal_actions())

def smart_strategy(state):
    decoded = decode_actions(state)
    if not decoded: return 0
    groups = {1: [], 2: [], 3: [], 4: []}
    for a, s in decoded:
        size = parse_combo_size(s)
        groups[size].append((a, s))
    for size in [4, 3, 2, 1]:
        if groups[size]:
            return min(groups[size], key=lambda x: parse_rank(x[1]))[0]
    return 0

strategy_map = {
    "random": random_action_strategy,
    "max_combo": max_combo_strategy,
    "single_only": single_only_strategy,
    "smart": smart_strategy,
    "aggressive": aggressive_strategy
}

# === 4️⃣ Agenten vorbereiten ===
agents = []
for pid, ptype in enumerate(PLAYER_TYPES):
    if ptype == "ppo":
        agent = ppo.PPOAgent(
            env.observation_spec()["info_state"][0],
            env.action_spec()["num_actions"],
            ppo.DEFAULT_CONFIG
        )
        agents.append(agent)
    elif ptype in strategy_map:
        agents.append(strategy_map[ptype])
    else:
        raise ValueError(f"Unbekannter Spielertyp: {ptype}")

# === 5️⃣ Ordnerstruktur ===
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

print(f"✅ Training gestartet. Version: {version_name} → {PLAYER_TYPES}")

# === 6️⃣ Eval-Funktion ===
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
        if np.argmax(time_step.rewards) == 0:
            wins += 1
    return wins / num_games

# === 7️⃣ Training ===
returns = [[] for _ in range(4)]
win_rates = []
eval_winrates = []
win_counts = defaultdict(int)
start_counts = defaultdict(int)
original_reward_sums = defaultdict(float)
shaped_reward_sums = defaultdict(float)

progress = trange(1, NUM_EPISODES + 1, desc="Training", unit="ep")

for ep in progress:
    time_step = env.reset()
    start_counts[time_step.observations["current_player"]] += 1
    steps = 0

    while not time_step.last():
        p = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][p]
        agent = agents[p]
        if isinstance(agent, ppo.PPOAgent):
            out = agent.step(time_step, legal)
            action = out.action if out else np.random.choice(legal)
        else:
            state = env.get_state
            action = agent(state)
        time_step = env.step([action])
        steps += 1
        if steps > 200:
            progress.write(f"⚠️ Episode {ep} abgebrochen bei 200 Schritten")
            break

    # Original & Shaped Rewards
    original_rewards = list(time_step.rewards)
    for pid, reward in enumerate(original_rewards):
        original_reward_sums[pid] += reward

    # Ranking → shaped rewards
    ranking = np.argsort(-np.array(original_rewards))
    shaped_rewards = [0] * 4
    for i, pid in enumerate(ranking):
        if i < len(REWARD_MAPPING):
            shaped_rewards[pid] = REWARD_MAPPING[i]

    winner = np.argmax(original_rewards)
    win_counts[winner] += 1
    win_rates.append(1 if winner == 0 else 0)

    for pid in range(4):
        shaped_reward_sums[pid] += shaped_rewards[pid]
        if isinstance(agents[pid], ppo.PPOAgent):
            agents[pid].step(time_step, [0])
        returns[pid].append(shaped_rewards[pid])

    if ep % 100 == 0:
        avg_ret = [np.mean(r[-100:]) for r in returns]
        win_avg = np.mean(win_rates[-100:]) * 100
        msg = " | ".join([f"P{pid}: {ret:.2f}" for pid, ret in enumerate(avg_ret)])
        progress.write(f"[Ep {ep}] Ø Return: {msg} | Winrate P0: {win_avg:.1f}%")

    if ep % 500 == 0 and isinstance(agents[0], ppo.PPOAgent):
        eval_wr = evaluate_against_random(env, agents[0])
        eval_winrates.append((ep, eval_wr))
        progress.write(f"[Eval] Ep {ep}: P0 vs Random Winrate: {eval_wr*100:.1f}%")

    if ep % 1000 == 0:
        for pid, agent in enumerate(agents):
            if isinstance(agent, ppo.PPOAgent):
                checkpoint_path = os.path.join(train_dir, f"checkpoint_ep{ep}_p{pid}")
                agent.save(checkpoint_path)

# === 8️⃣ Speichern ===
for pid, agent in enumerate(agents):
    if isinstance(agent, ppo.PPOAgent):
        agent.save(os.path.join(train_dir, f"{version_name}_agent_p{pid}"))
        np.save(os.path.join(train_dir, f"returns_p{pid}.npy"), returns[pid])

np.save(os.path.join(train_dir, "win_rates.npy"), win_rates)
for f in os.listdir(train_dir):
    shutil.copy(os.path.join(train_dir, f), os.path.join(test_dir, f))

# === 🔟 Plots ===
if GENERATE_PLOTS:
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
    path = os.path.join(train_dir, "ppo_winrate_over_time.png")
    plt.savefig(path)
    print(f"📊 Winrate-Plot gespeichert: {path}")
    plt.show()

    if eval_winrates:
        eps, scores = zip(*eval_winrates)
        plt.figure(figsize=(12,6))
        plt.plot(eps, scores, marker='o')
        plt.title(f"{version_name}: P0 vs Random Agents")
        plt.xlabel("Training Episode")
        plt.ylabel("Winrate P0")
        plt.ylim(-0.05, 1.05)
        plt.grid(True)
        path = os.path.join(train_dir, "eval_vs_random_plot.png")
        plt.savefig(path)
        print(f"📊 Eval-Plot gespeichert: {path}")
        plt.show()

# === 🔁 Statistiken ===
print("\n=== Startspieler-Statistik ===")
total_starts = sum(start_counts.values())
for pid in range(4):
    count = start_counts[pid]
    share = 100 * count / total_starts if total_starts else 0
    print(f"Player {pid} started {count} times ({share:.2f}%)")

print("\n=== Siegstatistik ===")
total_wins = sum(win_counts.values())
for pid in range(4):
    count = win_counts[pid]
    share = 100 * count / total_wins if total_wins else 0
    print(f"Player {pid} ({PLAYER_TYPES[pid]}) won {count} games ({share:.2f}%)")

print("\n=== Rewards ===")
for pid in range(4):
    print(f"Player {pid} ({PLAYER_TYPES[pid]}) → Original: {original_reward_sums[pid]:.1f} | Shaped: {shaped_reward_sums[pid]:.1f}")
