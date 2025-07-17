import os
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import ppo_local as ppo

NUM_EPISODES_PER_POSITION = 2500
MODEL_DIR = "Playground/models/selfplay_president_03/train"

# === Heuristik-Strategien ===
def max_combo(state):
    player = state.current_player()
    decoded = [(a, state.action_to_string(player, a)) for a in state.legal_actions() if a != 0]
    if not decoded:
        return 0
    def combo_key(s):
        parts = s.split()
        if parts[1] == "Single": return 1
        if parts[1] == "Pair": return 2
        if parts[1] == "Triple": return 3
        if parts[1] == "Quad": return 4
        return 0
    return max(decoded, key=lambda x: combo_key(x[1]))[0]

def random_action(state):
    return np.random.choice(state.legal_actions())

# === Agent-Typen konfigurieren ===
# Jeder Eintrag ist ein Agent-Typ, den du testen willst
# 'ppo' lädt ein trainiertes Modell, 'random' oder 'max_combo' sind feste Strategien
agent_types = ["ppo", "random", "random", "random"]

# === Lade Spielumgebung ===
game = pyspiel.load_game("president", {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False,
})
env = rl_environment.Environment(game)

# === Statistik ===
stats = {pid: {"returns": [], "wins": [], "roles": []} for pid in range(4)}

# === Hauptloop: Jeder Agent-Typ einmal auf jeder Position ===
for rotation in range(4):
    print(f"\n🔁 Rotation {rotation + 1}/4")

    # Agenten für diese Rotation zusammenbauen
    agents = []
    for pid in range(4):
        role = agent_types[(pid + rotation) % 4]
        stats[pid]["roles"].append(role)
        if role == "ppo":
            agent = ppo.PPOAgent(
                env.observation_spec()["info_state"][0],
                env.action_spec()["num_actions"],
                ppo.DEFAULT_CONFIG
            )
            path = os.path.join(MODEL_DIR, f"selfplay_president_03_agent_p{(pid + rotation)%4}")
            agent.restore(path)
            agents.append(agent)
        elif role == "max_combo":
            agents.append(max_combo)
        else:
            agents.append(random_action)

    # Spiele starten
    for ep in range(NUM_EPISODES_PER_POSITION):
        time_step = env.reset()
        state = env.get_state

        while not time_step.last():
            p = time_step.observations["current_player"]
            if callable(agents[p]):
                action = agents[p](env.get_state)
            else:
                obs = time_step.observations["info_state"][p]
                legal = time_step.observations["legal_actions"][p]
                out = agents[p].step(time_step, legal)
                action = out.action if out else np.random.choice(legal)
            time_step = env.step([action])

        # Werte speichern
        final_rewards = time_step.rewards
        winner = np.argmax(final_rewards)
        for pid in range(4):
            stats[pid]["returns"].append(final_rewards[pid])
            stats[pid]["wins"].append(1 if pid == winner else 0)

# === Ergebnis ausgeben ===
print("\n=== Gesamt-Ergebnisse nach Rotation ===\n")
for pid in range(4):
    role_list = stats[pid]["roles"]
    unique_roles = list(set(role_list))
    role_str = ", ".join(unique_roles)
    avg_return = np.mean(stats[pid]["returns"])
    winrate = np.mean(stats[pid]["wins"]) * 100
    print(f"Player {pid}:")
    print(f"  - Rollen: {role_str}")
    print(f"  - Ø Return: {avg_return:.2f}")
    print(f"  - Winrate: {winrate:.1f}%\n")
