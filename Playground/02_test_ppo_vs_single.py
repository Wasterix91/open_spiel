import os
import pyspiel
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

"""
Dieses Programm lädt eine zuvor trainierte Policy (Neurales Netzwerk) für das Kartenspiel 
'President' (1v1) und führt n (NUM_GAMES) Testspiele gegen einen fix programmierten Gegner aus, 
der stets die kleinste mögliche Karte spielt. Dabei wird die Winrate des gelernten Agenten 
berechnet und alle Ergebnisse in einer Log-Datei gespeichert.
"""


# === 📁 Basis: immer relativ zum Skript
base_dir = os.path.dirname(__file__)
models_root = os.path.join(base_dir, "models")

# === 🏷️ Manuelle Wahl: Version & Ordner
version = "ppo_president_08"   # <--- Deine Version hier
subfolder = "test"            # <--- 'train' oder 'test'

MODEL_DIR = os.path.join(models_root, version, subfolder)
LOG_FILE = os.path.join(MODEL_DIR, "policy_log.txt")

# === Korrekte Dateinamen automatisch
policy_file = f"{version}_policy.pt"
policy_path = os.path.join(MODEL_DIR, policy_file)
if not os.path.exists(policy_path):
    raise FileNotFoundError(f"❌ Policy-Datei nicht gefunden: {policy_path}")

print(f"✅ Lade Version: {version}/{subfolder}")
print(f"📁 Pfad: {MODEL_DIR}")

# === 🎮 Spiel erstellen (2 Spieler!)
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
        "num_players": 2  # <-- 1v1 Modus
    }
)

# === 🧠 Gelernte Policy laden
info_state_size = game.observation_tensor_shape()[0]
num_actions = game.num_distinct_actions()

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

policy_p0 = PolicyNetwork(info_state_size, num_actions)
policy_p0.load_state_dict(torch.load(policy_path))
policy_p0.eval()

print(f"✅ Gelernte Policy geladen: {policy_file}")

# === 🤖 Fixer 1v1-Gegner: Immer kleinste Karte
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s): return RANK_TO_NUM[s.split()[-1]]

def safe_choose(player, state):
    assert state.current_player() == player
    actions = state.legal_actions()
    if len(actions) == 1:
        return actions[0]
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]
    if not decoded:
        return 0
    return min(decoded, key=lambda x: parse_rank(x[1]))[0]

# === 🎬 Testspiele ausführen
NUM_GAMES = 100
all_returns = []
winner_counter = Counter()

with open(LOG_FILE, "w") as log:
    for g in range(1, NUM_GAMES + 1):
        state = game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            legal_actions = state.legal_actions()

            if player == 0:
                obs = state.information_state_tensor(player)
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                logits = policy_p0(obs_tensor).detach().numpy()

                masked_logits = np.zeros_like(logits)
                masked_logits[legal_actions] = logits[legal_actions]
                if masked_logits.sum() == 0:
                    probs = np.zeros_like(logits)
                    probs[legal_actions] = 1.0 / len(legal_actions)
                else:
                    probs = masked_logits / masked_logits.sum()

                action = np.random.choice(len(probs), p=probs)
            else:
                action = safe_choose(player, state)

            state.apply_action(action)

        returns = state.returns()
        all_returns.append(returns[0])
        winner = np.argmax(returns)
        winner_counter[winner] += 1

        log.write(f"Game {g}: PPO Agent Return = {returns[0]}, Winner = P{winner}\n")

print(f"✅ Alle {NUM_GAMES} 1v1-Spiele abgeschlossen. Log: {LOG_FILE}")

# === ✅ Finale Auswertung
binary_returns = [1 if r > 0 else 0 for r in all_returns]
wins = sum(binary_returns)
losses = NUM_GAMES - wins
winrate = wins / NUM_GAMES * 100

print(f"\n📊 Ergebnis für {NUM_GAMES} Spiele:")
print(f"  ➤ Siege: {wins}")
print(f"  ➤ Niederlagen: {losses}")
print(f"  ➤ Winrate: {winrate:.2f} %")

print("\n🏆 Gewinner gesamt:")
for player, count in sorted(winner_counter.items()):
    print(f"  ➤ Spieler P{player}: {count} Siege")

# === Alles auch im Log festhalten
with open(LOG_FILE, "a") as log:
    log.write(f"\n✅ Final: {wins} Wins / {losses} Losses\n")
    log.write(f"Winrate: {winrate:.2f} %\n")
    log.write("\n🏆 Winner counts:\n")
    for player, count in sorted(winner_counter.items()):
        log.write(f"  ➤ P{player}: {count} Wins\n")
