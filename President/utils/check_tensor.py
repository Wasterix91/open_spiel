import pyspiel

# Lade dein Game (Parameter bei Bedarf anpassen)
game = pyspiel.load_game("president", {"num_players": 4, "shuffle_cards": True})
state = game.new_initial_state()

def parse_obs(obs_str, nplayers):
    parts = [int(x) for x in obs_str.split(",")]
    k_num_ranks = len(parts) - (nplayers - 1) - 3
    last_rel = parts[k_num_ranks + (nplayers - 1) + 0]
    combo    = parts[k_num_ranks + (nplayers - 1) + 1]
    top_rank = parts[k_num_ranks + (nplayers - 1) + 2]
    return last_rel, combo, top_rank

# Ein paar Züge spielen und prüfen:
for step in range(20):
    # Einfach irgendeine legale Aktion nehmen (falls nur Pass möglich, wird auch das geprüft)
    actions = state.legal_actions()
    if not actions or state.is_terminal():
        break
    state.apply_action(actions[0])

    # Prüfe für alle Spieler die Masking-Logik: last_rel==0 => combo==0 & top==-1
    for p in range(game.num_players()):
        last_rel, combo, top = parse_obs(state.observation_string(p), game.num_players())
        assert 0 <= last_rel < game.num_players(), "last_played_relative außerhalb 0..n-1"
        if last_rel == 0:
            assert combo == 0 and top == -1, "Masking verletzt bei neuem Trick"
        else:
            assert combo > 0 and top >= 0, "Laufender Trick, aber combo/top ungültig"

print("OK: last_played_relative & Masking sind konsistent.")
