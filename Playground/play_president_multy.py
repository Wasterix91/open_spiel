import pyspiel
from collections import Counter
import random

# === 1️⃣ Einstellungen ===
NUM_SIMULATIONS = 100
NUM_EXAMPLES = 5  # Wie viele zufällig zeigen

# === 2️⃣ Spiel-Parameter ===
GAME_PARAMS = {
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
    "start_player_mode": "loser"
}

# === 3️⃣ Ranks dynamisch ===
def get_ranks(deck_size):
    if deck_size == "32":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "52":
        return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "64":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    else:
        raise ValueError(f"Unknown deck_size: {deck_size}")

RANKS = get_ranks(GAME_PARAMS["deck_size"])
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s):
    return RANK_TO_NUM[s.split()[-1]]

def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1

# === 4️⃣ Spieler-Strategien ===
def choose_action(state):
    player = state.current_player()
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if player == 0:
        if decoded:
            best = max(decoded, key=lambda x: parse_rank(x[1]))
            return best[0]
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
    return 0

# === 5️⃣ 100 Spiele simulieren ===
results = []  # Gewinner speichern
examples = []  # Raw Log-Daten von ausgewählten Beispielen

# Wähle zufällig aus, welche Indizes gezeigt werden
example_indices = set(random.sample(range(NUM_SIMULATIONS), NUM_EXAMPLES))

for sim in range(NUM_SIMULATIONS):
    game = pyspiel.load_game("president", GAME_PARAMS)
    state = game.new_initial_state()

    log = []  # Nur für dieses Spiel

    while not state.is_terminal():
        player = state.current_player()
        actions = state.legal_actions()
        chosen = choose_action(state)
        log.append({
            "player": player,
            "action": state.action_to_string(player, chosen),
            "state": str(state)
        })
        state.apply_action(chosen)

    # Returns speichern
    returns = state.returns()
    winner = max(range(len(returns)), key=lambda p: returns[p])
    results.append(winner)

    # Dieses Spiel merken, wenn es zu den Beispielen gehört
    if sim in example_indices:
        examples.append({
            "index": sim,
            "log": log,
            "returns": returns
        })

# === 6️⃣ Auswertung ===
counter = Counter(results)

print(f"\n=== Zusammenfassung nach {NUM_SIMULATIONS} Spielen ===")
for player in range(4):
    print(f"Player {player} gewann {counter[player]} mal ({counter[player]/NUM_SIMULATIONS:.1%})")

# === 7️⃣ Zeige 5 zufällige Beispiel-Spiele ===
print(f"\n=== {NUM_EXAMPLES} zufällige Beispiel-Spiele ===")

for example in examples:
    print(f"\n--- Beispiel Spiel {example['index']} ---")
    for step, entry in enumerate(example['log'], start=1):
        print(f"Runde {step}: Player {entry['player']} wählt {entry['action']}")
    print(f"Returns: {example['returns']}")
