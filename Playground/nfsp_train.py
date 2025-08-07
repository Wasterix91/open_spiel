import os
import re
import numpy as np
import pyspiel
import torch
import matplotlib.pyplot as plt
import pandas as pd
from open_spiel.python import rl_environment
from nfsp_agent import NFSPAgent

# === Hilfsfunktionen ===
def find_next_version(base_dir="models"):
    pattern = re.compile(r"nfsp_model_(\d{2})$")
    existing = [
        int(m.group(1))
        for m in (
            pattern.match(name)
            for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))
        )
        if m
    ]
    return f"{max(existing) + 1:02d}" if existing else "01"

# === Speicherort vorbereiten ===
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
VERSION = find_next_version(MODELS_ROOT)
MODEL_BASE = os.path.join(MODELS_ROOT, f"nfsp_model_{VERSION}", "train")
os.makedirs(MODEL_BASE, exist_ok=True)
print(f"📁 Neue Trainingsversion: {VERSION}")

# === Trainingsparameter ===
NUM_EPISODES = 60000
EVAL_INTERVAL = 1000
EVAL_EPISODES = 10000
NUM_PLAYERS = 4
ANTICIPATORY_PARAM = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Spiel & Environment ===
game = pyspiel.load_game("president", {
    "num_players": NUM_PLAYERS,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False
})
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === Agenten initialisieren ===
agents = [
    NFSPAgent(
        state_size=info_state_size,
        num_actions=num_actions,
        anticipatory_param=ANTICIPATORY_PARAM,
        device=DEVICE
    ) for _ in range(NUM_PLAYERS)
]

# === Evaluationstrategie ===
def random2_strategy(state):
    legal = state.legal_actions()
    return np.random.choice([a for a in legal if a != 0]) if len(legal) > 1 and 0 in legal else np.random.choice(legal)

opponent_fn = random2_strategy
winrates, eval_steps = [], []

# === Trainingsschleife ===
for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()
    obs_states = [None] * NUM_PLAYERS
    actions = [None] * NUM_PLAYERS
    total_reward_p0 = 0

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]
        obs = time_step.observations["info_state"][player]

        action = agents[player].select_action(obs, legal)
        obs_states[player] = obs
        actions[player] = action
        time_step = env.step([action])

        next_obs = time_step.observations["info_state"]
        reward = 0.0
        if player == 0:
            hand_size = sum(next_obs[0][:8])
            reward = -hand_size
            total_reward_p0 += reward

        agents[player].observe_transition(
            obs_states[player], actions[player], reward,
            next_obs[player], time_step.last()
        )

    # Bonus für Platzierung
    final_scores = time_step.rewards
    if final_scores[0] == max(final_scores):
        total_reward_p0 += 10
    elif final_scores[0] == min(final_scores):
        total_reward_p0 -= 5

    agents[0].observe_transition(
        obs_states[0], actions[0], total_reward_p0,
        time_step.observations["info_state"][0], True
    )

    for agent in agents:
        agent.train_step()

    if episode % 1000 == 0:
        for agent in agents:
            agent.update_target_network()

    if episode % EVAL_INTERVAL == 0:
        print(f"✅ Episode {episode} abgeschlossen.")
        print(f"🧪 Evaluation nach {episode} Episoden...")
        wins = 0
        for _ in range(EVAL_EPISODES):
            state = game.new_initial_state()
            while not state.is_terminal():
                pid = state.current_player()
                legal = state.legal_actions(pid)
                if pid == 0:
                    obs = state.observation_tensor(pid)
                    action = agents[0].act_sl(obs, legal)
                else:
                    action = opponent_fn(state)
                state.apply_action(action)
            if state.returns()[0] == max(state.returns()):
                wins += 1
        winrate = 100 * wins / EVAL_EPISODES
        winrates.append(winrate)
        eval_steps.append(episode)
        print(f"✅ Winrate gegen random2: {winrate:.2f}%")

# === Lernkurve & Evaluation speichern ===
plt.figure(figsize=(10, 6))
plt.plot(eval_steps, winrates, marker="o")
plt.xlabel("Episode")
plt.ylabel("Winrate gegen random2")
plt.title("NFSP – Lernkurve gegen random2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_BASE, "lernkurve.png"))

df_eval = pd.DataFrame({
    "episode": eval_steps,
    "winrate_percent": winrates
})
df_eval.to_csv(os.path.join(MODEL_BASE, "eval_results.csv"), index=False)

# === Modelle speichern ===
for i, ag in enumerate(agents):
    ag.save(os.path.join(MODEL_BASE, f"nfsp_model_{VERSION}_agent_p{i}"))

print("🏁 Training abgeschlossen.")
