import pyspiel

"""
Dieses Programm erstellt ein President-Spiel mit der gewählten deck_size ("32", "52" oder "64") 
und simuliert ein einzelnes Spiel mit festen Heuristik-Strategien für alle vier Spieler.
Player 0 spielt stets die höchste Karte, Player 1 spielt die größte Kombo, Player 2 spielt defensiv 
(kleinste Kombo mit niedrigstem Rang) und Player 3 spielt nur Einzelkarten.
Es zeigt den Spielverlauf, gewählte Züge und den finalen Return im Terminal an.
"""


# === 1️⃣ Spiel erstellen ===
# Hier kannst du deck_size ändern: "32", "52" oder "64"
game = pyspiel.load_game(
    "president",
    {
        "deck_size": "32",
        "shuffle_cards": True,
        "single_card_mode": False,
        "num_players":2
    }
)

state = game.new_initial_state()

print("=== President Game ===")
print(f"Num players: {game.num_players()}")
print(f"Num distinct actions: {game.num_distinct_actions()}")
print(f"Observation tensor shape: {game.observation_tensor_shape()}")

params = game.get_parameters()
print(f"shuffle_cards: {params['shuffle_cards']}")
print(f"single_card_mode: {params['single_card_mode']}")

print(f"deck_size: {params['deck_size']}")

# === 2️⃣ Spielzustand anzeigen ===
print("\nInitial State:")
print(state)

# === 3️⃣ Ranks dynamisch basierend auf deck_size ===
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

# === 4️⃣ Strategie für jeden Spieler ===
def choose_action(state):
    player = state.current_player()
    actions = state.legal_actions()
    decoded = [(a, state.action_to_string(player, a)) for a in actions if a != 0]

    if player == 0:
        # Hoch stechen: höchste Rank erlaubt
        if decoded:
            best = max(decoded, key=lambda x: parse_rank(x[1]))
            return best[0]
    elif player == 1:
        # Viele Karten: größte Combo Size
        if decoded:
            best = max(decoded, key=lambda x: parse_combo_size(x[1]))
            return best[0]
    elif player == 2:
        # Defensiv: kleinste Combo & kleinste Rank
        if decoded:
            best = min(decoded, key=lambda x: (parse_combo_size(x[1]), parse_rank(x[1])))
            return best[0]
    elif player == 3:
        # Nur Einzelkarten: finde Single, sonst Pass
        singles = [x for x in decoded if "Single" in x[1]]
        if singles:
            best = min(singles, key=lambda x: parse_rank(x[1]))
            return best[0]

    # Kein passender Zug -> erste erlaubte Aktion
    return actions[0]

# === 5️⃣ Spiel ausführen ===
for move in range(300):
    if state.is_terminal():
        break

    player = state.current_player()
    actions = state.legal_actions()
    action_strs = [state.action_to_string(player, a) for a in actions]

    print(f"\n=== Runde {move + 1} ===")
    print(state)
    print(state.observation_tensor())
    print(f"Player {player} legal actions: {action_strs}")

    chosen = choose_action(state)
    print(f"Player {player} wählt: {state.action_to_string(player, chosen)}")

    state.apply_action(chosen)


if state.is_terminal():
    print()
    print("Spiel beendet.")
    print()
    print(state)
    print("Returns:", state.returns())
    #print(state.get_finish_order())
