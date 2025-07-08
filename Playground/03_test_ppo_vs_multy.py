import os
import pyspiel
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

"""
Dieses Programm lädt eine trainierte Policy für das Kartenspiel 'President' 
und führt 100 Testspiele gegen Heuristik-Gegner mit unterschiedlichen Strategien aus. 
Es berechnet statistische Kennzahlen, erstellt ein Histogramm der Returns 
und speichert alle Ergebnisse sowie den Plot im gewählten Modell-Ordner.
"""


# === 📁 Basis: immer relativ zum Skript
base_dir = os.path.dirname(__file__)
models_root = os.path.join(base_dir, "models")

# === 🏷️ Manuelle Wahl: Version & Ordner
version = "ppo_president_04"   # <--- Hier deine Version
subfolder = "train"            # <--- 'train' oder 'test'

MODEL_DIR = os.path.join(models_root, version, subfolder)
LOG_FILE = os.path.join(MODEL_DIR, "policy_log.txt")
PLOT_FILE = os.path.join(MODEL_DIR, "policy_returns_histogram.png")

# === 📁 Korrekte Dateinamen automatisch
policy_file = f"{version}_policy.pt"
value_file = f"{version}_value.pt"

# === ✅ Sicherheit: Existenz prüfen
policy_path = os.path.join(MODEL_DIR, policy_file)
if not os.path.exists(policy_path):
    raise FileNotFoundError(f"❌ Policy-Datei nicht gefunden: {policy_path}")

print(f"✅ Lade Version: {version}/{subfolder}")
print(f"📁 Pfad: {MODEL_DIR}")

# === 🎮 Spiel erstellen
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
    }
)

# === 🧠 Gelernte Policy laden
info_state_size = game.information_state_tensor_shape()[0]
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

# === 🤖 Heuristik-Gegner ===
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s): return RANK_TO_NUM[s.split()[-1]]
def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1

def safe_choose(player, state):
    # === Sicherheit: Stelle sicher, dass der aufrufende Spieler auch wirklich dran ist
    assert state.current_player() == player

    # === Alle aktuell erlaubten Aktionen für diesen Spieler abfragen
    actions = state.legal_actions()

    # === Wenn es nur EINE legale Aktion gibt, dann MUSS der Spieler diese wählen (meist "Pass")
    if len(actions) == 1:
        return actions[0]

    # === Alle Aktionen decodieren: (Index, Klartext)
    # UND Aktionen mit Index 0 (Pass) zunächst ausfiltern
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    # === Wenn KEINE anderen Züge außer Pass möglich sind:
    # --> Muss Pass gespielt werden (Aktion 0)
    if not decoded:
        return 0

    # === Heuristik-Regeln für verschiedene Gegner:
    if player == 1:
        # Gegner 1: Spielt die KOMBI mit der größten Kartenanzahl (z.B. Quad vor Triple)
        choice = max(decoded, key=lambda x: parse_combo_size(x[1]))[0]

    elif player == 2:
        # Gegner 2: Spielt die kleinste Kombo mit der niedrigsten Kartenstufe
        # (kleinste Kombo-Größe und darin niedrigster Rang)
        choice = min(decoded, key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1])))[0]

    elif player == 3:
        # Gegner 3: Bevorzugt einzelne Karten
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            # Hat er Singles? --> Spiele die mit dem niedrigsten Rang
            choice = min(singles, key=lambda x: parse_rank(x[1]))[0]
        else:
            # Wenn keine Singles: spiele die niedrigste Kombo
            choice = min(decoded, key=lambda x: parse_rank(x[1]))[0]

    else:
        # Fehlerfall: Nur Spieler 1–3 sind erlaubt
        raise ValueError(f"Invalid player {player}")

    # === Sicherheits-Check:
    # Falls die Heuristik doch Pass (= 0) gewählt hätte:
    # Erzwinge, dass stattdessen die kleinste Karte gespielt wird
    if choice == 0:
        choice = min(decoded, key=lambda x: parse_rank(x[1]))[0]

    return choice


# === 🎬 Testspiele ausführen ===
NUM_GAMES = 100
all_returns = []
winner_counter = Counter()

with open(LOG_FILE, "w") as log:
    for g in range(1, NUM_GAMES + 1):
        state = game.new_initial_state()
        step = 0

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
            step += 1

        returns = state.returns()
        all_returns.append(returns[0])

        # Wer ist Sieger?
        winner = np.argmax(returns)
        winner_counter[winner] += 1

        log.write(f"Game {g}: PPO Agent Return = {returns[0]}, All Returns = {returns}, Winner = P{winner}\n")

print(f"✅ Alle {NUM_GAMES} Spiele abgeschlossen. Log gespeichert: {LOG_FILE}")

# === 📊 Statistik & Histogramm ===
mean_return = np.mean(all_returns)
std_return = np.std(all_returns)

print(f"\n📈 Statistik über {NUM_GAMES} Spiele:")
print(f"  ➤ Durchschnitt: {mean_return:.2f}")
print(f"  ➤ Standardabweichung: {std_return:.2f}")

print("\n🏆 Anzahl Siege pro Spieler:")
for player, count in sorted(winner_counter.items()):
    print(f"  ➤ Spieler P{player}: {count} Siege")

plt.figure(figsize=(8,5))
plt.hist(all_returns, bins=[-0.5,0.5,1.5,2.5,3.5], edgecolor='black', rwidth=0.8)
plt.title(f"PPO President: Return Distribution over {NUM_GAMES} Games")
plt.xlabel("Return (Place in Game)")
plt.ylabel("Frequency")
plt.xticks([0,1,2,3])
plt.grid(axis='y')
plt.savefig(PLOT_FILE)
print(f"📊 Histogramm gespeichert: {PLOT_FILE}")
