import pyspiel
import random
from collections import defaultdict

# 🔧 Einstellungen
NUM_GAMES = 1             # Für Menschliches Spiel nur 1 Spiel sinnvoll
NUM_SHOWN_GAMES = 1

# 💡 Parameter für das President-Spiel
DEFAULT_GAME_PARAMS = {
    "shuffle_cards": True,
    "single_card_mode": False  # False = Mehrfachkarten erlaubt, True = nur Einzelkarten
}

def play_human_vs_random(game):
    state = game.new_initial_state()
    round_log = []
    round_counter = 1

    while not state.is_terminal():
        log_lines = []
        log_lines.append(f"=== Runde {round_counter} ===")
        log_lines.extend(state.to_string().splitlines())

        legal_actions = state.legal_actions()
        player = state.current_player()

        if player == 0:
            print("\n".join(log_lines))
            print("Deine Aktionen:")
            for aid in legal_actions:
                print(f"{aid}: {state.action_to_string(player, aid)}")
            while True:
                try:
                    action = int(input("👉 Deine Aktion wählen (ID eingeben): "))
                    if action in legal_actions:
                        break
                    else:
                        print("❌ Ungültige Eingabe. Bitte gültige ID wählen.")
                except ValueError:
                    print("❌ Bitte eine ganze Zahl eingeben.")
        else:
            action = random.choice(legal_actions)

        action_str = state.action_to_string(player, action)
        log_lines.append(f"🎮 Spieler {player} spielt: {action_str}\n")

        round_log.append("\n".join(log_lines))
        state.apply_action(action)
        round_counter += 1

    returns = state.returns()
    winner = returns.index(max(returns))
    return winner, returns, round_log

def run_human_game():
    print("📦 Starte menschliches Spiel gegen Zufallsspieler...")
    game = pyspiel.load_game("president", DEFAULT_GAME_PARAMS)
    winner, returns, log = play_human_vs_random(game)

    print("\n🎮 Spielverlauf:")
    for entry in log:
        print(entry)
    print("\n🏁 Endstand:", returns)
    print(f"🏆 Gewinner: Spieler {winner} {'(DU)' if winner == 0 else ''}")

if __name__ == "__main__":
    run_human_game()
