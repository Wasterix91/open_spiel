import pyspiel

# === 1️⃣ Einstellungen ===
NUM_GAMES = 3

GAME_PARAMS = {
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
    "start_player_mode": "loser"
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

# Einmal festlegen:
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

# === 3️⃣ Strategie für jeden Spieler ===
def choose_action(state):
    player = state.current_player()
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if player == 2:
        if decoded:
            best = max(decoded, key=lambda x: parse_rank(x[1]))
            return best[0]
    elif player == 1:
        if decoded:
            best = max(decoded, key=lambda x: parse_combo_size(x[1]))
            return best[0]
    elif player == 0:
        if decoded:
            best = min(decoded, key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1])))
            return best[0]
    elif player == 3:
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            best = min(singles, key=lambda x: parse_rank(x[1]))
            return best[0]

    return 0  # Pass

# === 4️⃣ EINMAL das Spiel laden ===
game = pyspiel.load_game("president", GAME_PARAMS)

print("=== President Game ===")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Observation tensor shape: {game.observation_tensor_shape()}")

params = game.get_parameters()
print(f"shuffle_cards: {params['shuffle_cards']}")
print(f"single_card_mode: {params['single_card_mode']}")
print(f"start_player_mode: {params['start_player_mode']}")
print(f"deck_size: {params['deck_size']}")

# === 5️⃣ Mehrfach-Spiele laufen lassen ===
for game_idx in range(NUM_GAMES):
    print(f"\n\n=====================")
    print(f"=== Spiel {game_idx + 1} ===")
    print("=====================")

    state = game.new_initial_state()

    print(f"\nStartspieler: Player {state.current_player()}\n")
    print("Initial State:")
    print(state)

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
