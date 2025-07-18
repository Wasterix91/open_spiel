import os
import re
import numpy as np
import pyspiel
from open_spiel.python import rl_environment
import ppo_local_2 as ppo
import torch

# === Trainingsparameter ===
NUM_EPISODES = 20_000
EVAL_INTERVAL = 500
EVAL_EPISODES = 50

# === Speicherpfad an Skript #4 anpassen ===
# === Automatische Versionserkennung ===
def find_next_version(base_dir="models"):
    pattern = re.compile(r"selfplay_president_(\d{2})$")
    existing = [
        int(m.group(1))
        for m in (
            pattern.match(name)
            for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))
        )
        if m
    ]
    return f"{max(existing) + 1:02d}" if existing else "01"

VERSION = find_next_version("models")
MODEL_BASE = os.path.join("models", f"selfplay_president_{VERSION}", "train")
MODEL_PATH = os.path.join(MODEL_BASE, f"selfplay_president_{VERSION}_agent_p0")
os.makedirs(MODEL_BASE, exist_ok=True)

print(f"📁 Neue Trainingsversion: {VERSION}")


# === Spiel und Environment ===
game = pyspiel.load_game("president", {
    "num_players": 4,
    "deck_size": "32",
    "shuffle_cards": True,
    "single_card_mode": False
})
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]

# === PPO-Agent für Player 0 ===
agent = ppo.PPOAgent(info_state_size, num_actions)

# === Trainingsschleife ===
for episode in range(1, NUM_EPISODES + 1):
    time_step = env.reset()
    total_reward = 0

    while not time_step.last():
        player = time_step.observations["current_player"]
        legal = time_step.observations["legal_actions"][player]

        if player == 0:
            agent_out = agent.step(time_step, legal)
            action = agent_out.action if agent_out else np.random.choice(legal)
        else:
            action = np.random.choice(legal)

        time_step = env.step([action])

        if player == 0:
            hand = time_step.observations["info_state"][0]
            hand_size = sum(hand[:8])  # 8 Ränge bei 32er Deck
            reward = -hand_size
            agent._buffer.rewards[-1] = reward
            total_reward += reward

    # Bonus für Endplatzierung
    final_scores = time_step.rewards
    player_score = final_scores[0]
    if player_score == max(final_scores):
        total_reward += 10
    elif player_score == min(final_scores):
        total_reward -= 5

    agent._buffer.rewards[-1] = total_reward
    agent.step(time_step, [0])
    agent.train()

    if episode % 100 == 0:
        print(f"[{episode}] Training abgeschlossen.")

    # === Evaluation gegen Random-Gegner ===
    if episode % EVAL_INTERVAL == 0:
        wins = 0
        for _ in range(EVAL_EPISODES):
            state = game.new_initial_state()
            while not state.is_terminal():
                pid = state.current_player()
                if pid == 0:
                    obs = state.information_state_tensor(0)
                    legal = state.legal_actions(0)
                    logits = agent._policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
                    masked = np.zeros_like(logits)
                    masked[legal] = logits[legal]

                    if masked.sum() == 0 or np.any(np.isnan(masked)):
                        probs = np.ones(len(logits))
                        probs[legal] = 1.0
                        probs /= probs.sum()
                    else:
                        probs = masked / masked.sum()

                    action = np.random.choice(len(probs), p=probs)

                else:
                    action = np.random.choice(state.legal_actions(pid))
                state.apply_action(action)
            if state.returns()[0] == max(state.returns()):
                wins += 1
        winrate = 100 * wins / EVAL_EPISODES
        print(f"✅ Evaluation nach {episode} Episoden: Winrate gegen Random = {winrate:.1f}%")

        agent.save(MODEL_PATH)

# === Finales Modell speichern ===
agent.save(MODEL_PATH)
print(f"✅ Finales Modell gespeichert unter: {MODEL_PATH}")
