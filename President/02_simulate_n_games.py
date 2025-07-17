import pyspiel
from collections import defaultdict

"""
Dieses Programm führt 10.000 President-Spiele mit vordefinierten Heuristik-Strategien für alle Spieler aus.
Player 0: Smart Strategy. (große Kombos bevorzugt)
Player 1: spielt IMMER so viele Karten wie erlaubt (max Combo).
Player 2: spielt DEFENSIV: niedrige Ranks & kleine Kombos.
Player 3: spielt NUR Einzelkarten (oder passt).
Für jedes Spiel werden Returns, Siege und Startspieler statistisch erfasst und am Ende als Zusammenfassung ausgegeben.
"""


# === 1️⃣ Parameter ===
NUM_GAMES = 1000

GAME_PARAMS = {
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False,
    "num_players": 6
    #"start_player_mode": "loser"
}

# === 2️⃣ Ranks dynamisch ===
def get_ranks(deck_size):
    if deck_size == "32":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "52":
        return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    elif deck_size == "64":
        return ["7", "8", "9", "10", "J", "Q", "K", "A"]
    else:
        raise ValueError(f"Unknown deck_size: {deck_size}")

RANKS = get_ranks(GAME_PARAMS['deck_size'])
RANK_TO_NUM = {rank: i for i, rank in enumerate(RANKS)}

def parse_rank(s):
    return RANK_TO_NUM[s.split()[-1]]

def parse_combo_size(s):
    if "Single" in s: return 1
    if "Pair" in s: return 2
    if "Triple" in s: return 3
    if "Quad" in s: return 4
    return 1

def smart_strat(state):
    """
    - Kombos bevorzugt
    - Hohe Ranks wenn nötig
    - Lieber passen als unsinnig stechen
    """

    player = state.current_player()
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if not decoded:
        return 0  # Pass

    # === 1️⃣ Gruppiere nach Combo-Size ===
    singles = []
    pairs = []
    triples = []
    quads = []
    for a, s in decoded:
        size = parse_combo_size(s)
        if size == 1:
            singles.append((a, s))
        elif size == 2:
            pairs.append((a, s))
        elif size == 3:
            triples.append((a, s))
        elif size == 4:
            quads.append((a, s))

    # === 2️⃣ Ziel: größte Combo bevorzugt ===
    for group in [quads, triples, pairs, singles]:
        if group:
            # Wähle die Combo mit kleinstem Rank innerhalb der größten Gruppe
            best = min(group, key=lambda x: parse_rank(x[1]))
            return best[0]

    # === 3️⃣ Fallback ===
    return 0  # Pass

# === 3️⃣ Strategie ===
def choose_action(state):
    player = state.current_player()

    # Test: P0 = Smart, andere wie bisher
    if player == 0:
        return smart_strat(state)

    # Rest wie bisher:
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if player == 1:
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

    return actions[0]


# === 4️⃣ Spiel einmal laden ===
game = pyspiel.load_game("president", GAME_PARAMS)

# === 5️⃣ Statistiken ===
total_returns = defaultdict(float)
total_wins = defaultdict(int)
total_starts = defaultdict(int)

for game_idx in range(NUM_GAMES):
    state = game.new_initial_state()

    # Zähle den Startspieler (== der letzte Verlierer)
    starter = state.current_player()
    total_starts[starter] += 1

    while not state.is_terminal():
        chosen = choose_action(state)
        state.apply_action(chosen)

    returns = state.returns()
    for player, ret in enumerate(returns):
        total_returns[player] += ret

    winner = max(range(len(returns)), key=lambda p: returns[p])
    total_wins[winner] += 1

    if (game_idx + 1) % 100 == 0:
        print(f"✅ {game_idx + 1} Spiele abgeschlossen...")

# === 6️⃣ Zusammenfassung ===
print(f"\n=== Zusammenfassung nach {NUM_GAMES} Spielen ===\n")

total_points = sum(total_returns.values())

for player in range(game.num_players()):
    points = total_returns[player]
    wins = total_wins[player]
    starts = total_starts[player]
    share = 100 * points / total_points if total_points else 0
    print(f"Player {player}:")
    print(f"  - Gesamtpunkte: {points:.1f}")
    print(f"  - Siege: {wins} ({wins/NUM_GAMES:.1%})")
    print(f"  - Starts: {starts} ({starts/NUM_GAMES:.1%})")
    print(f"  - Anteil an allen Punkten: {share:.1f}%")
    print()

print("=== Strategien ===")
print("Player 0: Smart Strategy.")
print("Player 1: spielt IMMER so viele Karten wie erlaubt (max Combo).")
print("Player 2: spielt DEFENSIV: niedrige Ranks & kleine Kombos.")
print("Player 3: spielt NUR Einzelkarten (oder passt).")
