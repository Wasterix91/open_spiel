import os
import re
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import Playground.ppo_local as ppo
from tqdm import trange
import matplotlib.pyplot as plt
import shutil

# === 1️⃣ Spiel: President mit nur 2 Spielern ===
game = pyspiel.load_game(
    "president",
    {
        "num_players": 4,   # <<< nur 2 Spieler: PPO-Agent + 1 Gegner
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
    }
)
env = rl_environment.Environment(game)

# === 2️⃣ PPO-Agent erstellen ===
agent = ppo.PPOAgent(
    env.observation_spec()["info_state"][0],
    env.action_spec()["num_actions"],
    ppo.DEFAULT_CONFIG
)

# === 3️⃣ Kartenränge für Heuristik
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s):
    return RANK_TO_NUM[s.split()[-1]]

# === 4️⃣ Einfache Heuristik: Immer kleinste Karte spielen
def safe_choose(player, state):
    assert state.current_player() == player
    actions = state.legal_actions()
    if len(actions) == 1:
        return actions[0]
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]
    if not decoded:
        return 0
    return min(decoded, key=lambda x: parse_rank(x[1]))[0]

# === 5️⃣ Versionierung & Ordner anlegen ===
base_dir = os.path.dirname(__file__)
models_root = os.path.join(base_dir, "models")

prefix = "ppo_president_"
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

print(f"✅ Training startet. Version: {version_name}")

# === 6️⃣ Training ===
num_episodes = 20000  # Tipp: 5k zum Testen, 10–20k für sehr stabil
returns = []
win_rates = []
progress = trange(1, num_episodes + 1, desc="Training", unit="episode")

for ep in progress:
    time_step = env.reset()
    state = env.get_state
    steps = 0

    while not time_step.last():
        p = time_step.observations["current_player"]

        if p == 0:
            agent_out = agent.step(
                time_step,
                time_step.observations["legal_actions"][0]
            )
            action = agent_out.action
        else:
            action = safe_choose(p, state)

        time_step = env.step([action])
        steps += 1

        if steps > 200:
            progress.write(f"⚠️ Episode {ep} abgebrochen bei 200 Schritten (Sicherheit)")
            break

    # Finaler PPO Step mit legal_actions als Dummy-Liste
    agent.step(time_step, time_step.observations["legal_actions"][0])
    returns.append(sum(time_step.rewards))

    winner = np.argmax(time_step.rewards)
    p0_win = int(winner == 0)
    win_rates.append(p0_win)

    if ep % 100 == 0:
        avg_return = np.mean(returns[-100:])
        avg_win = np.mean(win_rates[-100:]) * 100
        progress.write(f"[Ep {ep}] Ø Return: {avg_return:.2f} | Winrate (last 100): {avg_win:.1f}%")

# === 7️⃣ Speichern
agent.save(os.path.join(train_dir, version_name))
np.save(os.path.join(train_dir, "training_returns.npy"), returns)
np.save(os.path.join(train_dir, "win_rates.npy"), win_rates)

# Auch in 'test' kopieren
for f in [f"{version_name}_policy.pt", f"{version_name}_value.pt", "training_returns.npy", "win_rates.npy"]:
    shutil.copy(os.path.join(train_dir, f), os.path.join(test_dir, f))

print(f"\n✅ Modell gespeichert in: {train_dir} & {test_dir}")

# === 8️⃣ Winrate Plot
window = 100
smoothed = np.convolve(win_rates, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12,6))
plt.plot(win_rates, color='lightblue', label="Win (0/1)")
plt.plot(range(window-1, len(win_rates)), smoothed, color='orange', label=f"Moving Avg ({window})")
plt.title(f"{version_name}: PPO Agent Win Rate Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Win (1 = P0 wins)")
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True)
plot_path = os.path.join(train_dir, "ppo_winrate_over_time.png")
plt.savefig(plot_path)
print(f"📊 Winrate-Plot gespeichert: {plot_path}")
plt.show()
