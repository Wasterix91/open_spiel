import pyspiel
import random
from collections import defaultdict

# TODO: Verschiedene Startpositionen (Random, Sieger letzte Runde, durchlaufend)

# 🔧 Einstellungen
NUM_GAMES = 1000          # Wie viele Spiele simulieren?
NUM_SHOWN_GAMES = 1       # Wie viele davon vollständig anzeigen?

def play_random_game(game):
    state = game.new_initial_state()
    round_log = []
    round_counter = 1

    while not state.is_terminal():
        log_lines = []
        log_lines.append(f"=== Runde {round_counter} ===")
        log_lines.extend(state.to_string().splitlines())

        legal_actions = state.legal_actions()
        action = random.choice(legal_actions)
        player = state.current_player()
        action_str = state.action_to_string(player, action)
        log_lines.append(f"🎮 Spieler {player} spielt: {action_str}\n")

        round_log.append("\n".join(log_lines))
        state.apply_action(action)
        round_counter += 1

    returns = state.returns()
    winner = returns.index(max(returns))
    return winner, returns, round_log

def run_games_with_summary(game_name="president", num_games=NUM_GAMES, show_examples=NUM_SHOWN_GAMES, game_params={"shuffle_cards": True}):
    print(f"📦 Lade Spiel: {game_name}")
    if game_params is None:
        game_params = {}
    game = pyspiel.load_game(game_name, game_params)

    win_counter = defaultdict(int)
    score_total = defaultdict(float)
    all_returns = []
    all_logs = []

    for game_idx in range(num_games):
        winner, returns, log = play_random_game(game)
        win_counter[winner] += 1
        all_returns.append(returns)
        all_logs.append(log)

        for i, score in enumerate(returns):
            score_total[i] += score

    num_players = len(all_returns[0])

    shown = min(show_examples, num_games)
    print(f"\n🎥 Zeige {shown} zufällig ausgewählte Spiele:\n")
    sample_indices = random.sample(range(num_games), shown)
    for idx in sample_indices:
        print(f"\n🕹️ Spiel {idx + 1}")
        for entry in all_logs[idx]:
            print(entry)
        print("🏁 Punkte:", all_returns[idx])
        print("-" * 40)

    print(f"\n📊 Statistik über {num_games} Spiele:")
    for player in range(num_players):
        wins = win_counter.get(player, 0)
        total = score_total[player]
        avg = total / num_games
        winrate = 100 * wins / num_games
        print(f"  Spieler {player}: {wins} Siege ({winrate:.1f}%), {total:.1f} Gesamtpunkte, Ø {avg:.2f} Punkte/Spiel")


if __name__ == "__main__":
    run_games_with_summary()
