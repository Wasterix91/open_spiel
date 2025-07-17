import os
import pyspiel
import numpy as np
from open_spiel.python import rl_environment
import Playground.ppo_local as ppo
from tqdm import trange
import re

# === 1️⃣ Spiel & Env ===
game = pyspiel.load_game("president", {
    "deck_size": "32",
    "shuffle_cards": True,

})
env = rl_environment.Environment(game)

# PPO-Agent für Spieler 0
agent = ppo.PPOAgent(
    env.observation_spec()["info_state"][0],
    env.action_spec()["num_actions"],
    ppo.DEFAULT_CONFIG
)

# === Heuristik-Spieler ===
RANKS = ["7", "8", "9", "10", "J", "Q", "K", "A"]
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s):
    return RANK_TO_NUM[s.split()[-1]]

def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1

def safe_choose(player, state):
    # === Sicherheit: Stelle sicher, dass 'player' auch wirklich am Zug ist
    assert state.current_player() == player

    # === Alle aktuell erlaubten Aktionen abrufen (z.B. welche Karten dürfen gespielt werden)
    actions = state.legal_actions()

    # === Falls nur eine legale Aktion erlaubt ist (typisch: nur 'Pass'):
    # diese sofort zurückgeben, Spieler MUSS sie wählen.
    if len(actions) == 1:
        return actions[0]

    # === Alle legalen Aktionen in menschenlesbare Form übersetzen.
    # Pass (Index 0) wird hier ignoriert, damit nur echte Spielzüge betrachtet werden.
    decoded = [
        (a, state.action_to_string(player, a)) for a in actions if a != 0
    ]

    # === Falls KEINE anderen Züge außer Pass möglich sind:
    # => Spieler MUSS passen (Aktion 0).
    if not decoded:
        return 0

    # === Nun folgt die Gegner-Heuristik:
    # Jeder Gegner (1, 2, 3) hat eine eigene simple Strategie,
    # wie er aus den erlaubten Zügen wählt.

    if player == 1:
        # Gegner 1: Wählt die größte mögliche Kombo (z.B. Quad > Triple > Pair)
        choice = max(decoded, key=lambda x: parse_combo_size(x[1]))[0]

    elif player == 2:
        # Gegner 2: Spielt die kleinste mögliche Kombo UND
        # innerhalb dieser die Karte mit dem niedrigsten Rang.
        choice = min(
            decoded,
            key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1]))
        )[0]

    elif player == 3:
        # Gegner 3: Bevorzugt 'Single'-Züge (Einzelkarten).
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            # Hat Singles: spiele die mit dem niedrigsten Rang.
            choice = min(singles, key=lambda x: parse_rank(x[1]))[0]
        else:
            # Keine Singles: spiele generell die Karte mit dem niedrigsten Rang.
            choice = min(decoded, key=lambda x: parse_rank(x[1]))[0]

    else:
        # Fehlerfall: nur Gegner 1–3 sind erlaubt.
        raise ValueError(f"Invalid player {player}")

    # === Sicherheitsnetz: Falls die Heuristik versehentlich 'Pass' gewählt hat:
    # Zwinge stattdessen die niedrigste echte Karte zu spielen.
    if choice == 0:
        choice = min(decoded, key=lambda x: parse_rank(x[1]))[0]

    # === Fertig: Rückgabe der gewählten Aktion.
    return choice


# === 2️⃣ Training ===
num_episodes = 500
returns = []
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
            progress.write(f"⚠️ Episode {ep} aborted after 200 steps (safety)")
            break

    agent.step(time_step, [])
    returns.append(sum(time_step.rewards))

    if ep % 100 == 0:
        avg = np.mean(returns[-100:])
        progress.write(f"[Episode {ep}] Ø Return (last 100): {avg:.2f}")



# === 📁 1️⃣ Basis-Pfad
base_dir = os.path.dirname(__file__)
models_root = os.path.join(base_dir, "models")

# === 📁 2️⃣ Alle vorhandenen Versionen suchen
prefix = "ppo_president_"
existing = [
    name for name in os.listdir(models_root)
    if os.path.isdir(os.path.join(models_root, name)) and re.match(rf"{prefix}\d+", name)
]

# === 📁 3️⃣ Höchste Version bestimmen
numbers = [int(re.findall(r"\d+", name)[0]) for name in existing] if existing else [0]
next_num = max(numbers) + 1

# Format mit führender Null: 01, 02, ...
new_folder_name = f"{prefix}{next_num:02d}"

# === 📁 4️⃣ Neues Unterverzeichnis 'train' in neuer Version
output_dir = os.path.join(models_root, new_folder_name, "train")
os.makedirs(output_dir, exist_ok=True)

# === ✅ 5️⃣ Speichern
agent.save(os.path.join(output_dir, new_folder_name))
np.save(os.path.join(output_dir, "training_returns.npy"), returns)

progress.write(f"\n✅ Alles gespeichert in: {output_dir}/")
progress.write(f" - Policy: {new_folder_name}_policy.pt")
progress.write(f" - Value:  {new_folder_name}_value.pt")
progress.write(f" - Returns: training_returns.npy")


