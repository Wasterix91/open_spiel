import random
from open_spiel.python import rl_environment
import pyspiel

# ---------------------------
# Regelbasierter Agent
# ---------------------------
class RuleBasedAgent:
    def __init__(self, player_id, pass_probability=0.0):
        self.player_id = player_id
        self.pass_probability = pass_probability

    def step(self, time_step):
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        if len(legal_actions) == 1:
            return {"action": legal_actions[0]}
        if random.random() < self.pass_probability:
            return {"action": 0} if 0 in legal_actions else {"action": random.choice(legal_actions)}
        else:
            # spiele stärkste Karte, falls nicht gepasst wird
            non_pass = [a for a in legal_actions if a != 0]
            return {"action": max(non_pass)}

# ---------------------------
# Simulation
# ---------------------------
def simulate_games(num_games=10, pass_probs=[0.1, 0.5, 0.5, 0.1], verbose=False):
    # Spiel mit RL-kompatibler Observation
    game = pyspiel.load_game("president(single_card_mode=True,shuffle_cards=True,start_player_mode=random)")
    env = rl_environment.Environment(game, include_full_state=True)

    agents = [RuleBasedAgent(i, pass_probs[i]) for i in range(4)]
    win_counter = [0] * 4
    all_returns = []

    random_game = None
    random_game_index = random.randint(0, num_games - 1)

    for episode in range(num_games):
        time_step = env.reset()
        traj = []

        while not time_step.last():
            pid = time_step.observations["current_player"]
            action = agents[pid].step(time_step)["action"]
            traj.append((pid, action))
            time_step = env.step([action])

        returns = time_step.rewards
        all_returns.append(returns)
        winner = returns.index(max(returns))
        win_counter[winner] += 1

        if episode == random_game_index:
            random_game = traj

    # Statistik
    print("\n--- Ergebnisse nach", num_games, "Spielen ---")
    for i, wins in enumerate(win_counter):
        print(f"Spieler {i}: {wins}x gewonnen (Aggressivität = {1 - pass_probs[i]:.2f})")

    # Beispielpartie anzeigen
    print("\n--- Beispielpartie (zufällig ausgewählt) ---")
    for pid, action in random_game:
        decoded = game.action_to_string(pid, action)
        print(f"Spieler {pid} spielt: {decoded}")

# ---------------------------
# Einstiegspunkt
# ---------------------------
if __name__ == "__main__":
    # Beispiel: Spieler 0 & 3 sind aggressiv, Spieler 1 & 2 defensiv
    simulate_games(
        num_games=10,
        pass_probs=[0.0, 0.5, 0.5, 0.0],  # Aggressivität durch Passwahrscheinlichkeit
        verbose=True
    )
