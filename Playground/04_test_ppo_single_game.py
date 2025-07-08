"""
Dieses Programm lädt eine trainierte Policy für Player 0 im Kartenspiel 'President' aus dem 'train'-Ordner und führt ein einzelnes Spiel gegen Heuristik-Gegner aus.
Es zeigt die Policy-Wahrscheinlichkeiten bei jedem Zug, loggt sie in eine Datei und erstellt am Ende einen kumulierten Verteilungsplot aller gewählten Aktionen, gespeichert im 'test'-Ordner.
"""

import os
import pyspiel
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# === 🏷️ Manuelle Wahl: Version
version = "ppo_president_02"   # <--- Deine Version hier

# === 📁 Basis: immer relativ zum Skript
base_dir = os.path.dirname(__file__)
models_root = os.path.join(base_dir, "models")

# === 🔄 Lade- und Speicherordner
train_dir = os.path.join(models_root, version, "train")
test_dir = os.path.join(models_root, version, "test")

# === ✅ Policy-Datei laden aus TRAIN
policy_file = f"{version}_policy.pt"
policy_path = os.path.join(train_dir, policy_file)

if not os.path.exists(policy_path):
    raise FileNotFoundError(f"❌ Policy-Datei nicht gefunden: {policy_path}")

print(f"✅ Lade Policy aus: {policy_path}")

# === ✅ Speicherpfade für LOG und PLOT im TEST Ordner
LOG_FILE = os.path.join(test_dir, "policy_log.txt")
PLOT_FILE = os.path.join(test_dir, "policy_distribution.png")

# === 🔒 Sicherstellen, dass TEST-Ordner existiert
os.makedirs(test_dir, exist_ok=True)

# === 1️⃣ Spiel erstellen ===
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
    }
)

state = game.new_initial_state()

print("\n=== President Game ===")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Observation tensor shape: {game.observation_tensor_shape()}")

params = game.get_parameters()
print(f"shuffle_cards: {params['shuffle_cards']}")
print(f"single_card_mode: {params['single_card_mode']}")
#print(f"start_player_mode: {params['start_player_mode']}")
print(f"deck_size: {params['deck_size']}")

# === 2️⃣ Ranks dynamisch basierend auf deck_size ===
def get_ranks(deck_size):
    if deck_size == "32":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "52":
        return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "64":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    else:
        raise ValueError(f"Unknown deck_size: {deck_size}")

RANKS = get_ranks(params['deck_size'])
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s):
    return RANK_TO_NUM[s.split()[-1]]

def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1  # fallback

# === 3️⃣ PolicyNetwork definieren ===
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

# === 4️⃣ Gelernte Policy laden ===
info_state_size = game.information_state_tensor_shape()[0]
num_actions = game.num_distinct_actions()

policy_p0 = PolicyNetwork(info_state_size, num_actions)
policy_p0.load_state_dict(torch.load(policy_path))
policy_p0.eval()

print("\n✅ Gelernte Policy für Player 0 geladen.")

# === 5️⃣ Statistik für Visualisierung ===
policy_counts = np.zeros(num_actions)

# === 6️⃣ Strategie für jeden Spieler ===
def choose_action(state):
    global policy_counts

    player = state.current_player()
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if player == 0:
        # === PPO-Policy wählen ===
        obs = state.information_state_tensor(player)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = policy_p0(obs_tensor).detach().numpy().flatten()

        # Maskieren auf legal actions
        masked_logits = np.zeros_like(logits)
        masked_logits[actions] = logits[actions]

        if masked_logits.sum() == 0:
            probs = np.zeros_like(logits)
            probs[actions] = 1.0 / len(actions)
        else:
            probs = masked_logits / masked_logits.sum()

        # 📊 Zeige Wahrscheinlichkeiten
        print(f"\n📊 Policy Wahrscheinlichkeiten für Player 0:")
        for a in actions:
            s = state.action_to_string(player, a)
            print(f"  {s:20s} : {probs[a]:.3f}")

        # Log in Datei
        with open(LOG_FILE, "a") as f:
            f.write(f"Player 0 Wahrscheinlichkeiten:\n")
            for a in actions:
                s = state.action_to_string(player, a)
                f.write(f"  {s:20s} : {probs[a]:.3f}\n")
            f.write("\n")

        # Zähle für Gesamt-Histogramm
        policy_counts += probs

        chosen = np.random.choice(len(probs), p=probs)
        return chosen

    elif player == 1:
        if decoded:
            best = max(decoded, key=lambda x: parse_combo_size(x[1]))
            return best[0]
    elif player == 2:
        if decoded:
            best = min(decoded, key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1])))
            return best[0]
    elif player == 3:
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            best = min(singles, key=lambda x: parse_rank(x[1]))
            return best[0]

    return 0  # kein passender Zug -> Pass

# === 7️⃣ Spiel ausführen ===
for move in range(100):
    if state.is_terminal():
        print("\nSpiel ist vorbei.")
        break

    player = state.current_player()
    actions = state.legal_actions()
    action_strs = [state.action_to_string(player, a) for a in actions]

    print(f"\n=== Runde {move + 1} ===")
    print(f"Player {player} legal actions: {action_strs}")

    chosen = choose_action(state)
    print(f"Player {player} wählt: {state.action_to_string(player, chosen)}")

    state.apply_action(chosen)
    print(state)

if state.is_terminal():
    print("\nSpiel beendet. Returns:", state.returns())

# === 8️⃣ Am Ende: Plot erstellen ===
print("\n📈 Erstelle kumulierte Policy-Verteilung für Player 0")

plt.figure(figsize=(12, 4))
plt.bar(range(num_actions), policy_counts)
plt.xlabel("Action ID")
plt.ylabel("Kumulierte Wahrscheinlichkeit (über alle Züge)")
plt.title("Policy-Verteilung von Player 0 über alle Züge")
plt.savefig(PLOT_FILE)
print(f"✅ Plot gespeichert: {PLOT_FILE}")
