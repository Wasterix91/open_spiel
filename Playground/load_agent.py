from ppo_agent import PPOAgent
import pyspiel
from open_spiel.python import rl_environment

game_settings = {
    "num_players": 4,
    "deck_size": "64",
    "shuffle_cards": True,
    "single_card_mode": False
}
game = pyspiel.load_game("president", game_settings)
env = rl_environment.Environment(game)
info_state_size = env.observation_spec()["info_state"][0]
num_actions = env.action_spec()["num_actions"]
ag = PPOAgent(info_state_size, num_actions)

ag.restore("models/ppo_model_34/train/ppo_model_34_agent_p0_ep0030000")

print(list(ag._policy.parameters()))
#ag.step(...)