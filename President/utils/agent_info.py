# open_spiel/President/utils/agent_plot.py
from pathlib import Path
from shutil import which
import argparse

from torchinfo import summary
from torchview import draw_graph
from agents.ppo_agent import PPOAgent
import pyspiel
from open_spiel.python import rl_environment


def render_torchview(module, name: str, input_dim: int, out_dir: Path):
    """Rendert mit torchview; speichert PNG in utils/agent_plots (oder DOT, falls 'dot' fehlt)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = draw_graph(module, input_size=(1, input_dim), graph_name=name, device="cpu")
    if which("dot"):
        g.visual_graph.render(
            filename=f"{name}_layers",
            directory=str(out_dir),
            format="png",
            cleanup=True,
        )
        print(f"✅ geschrieben: {out_dir / f'{name}_layers.png'}")
    else:
        dot_path = out_dir / f"{name}.dot"
        dot_path.write_text(g.visual_graph.source)
        print(f"⚠️  Graphviz 'dot' nicht gefunden. DOT exportiert nach: {dot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-prefix",
        default="models/ppo_model_06/train/ppo_model_06_agent_p0_ep0014000",
        help="Pfadpräfix zu *_policy.pt und *_value.pt (ohne Suffix)",
    )
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--deck-size", type=str, default="64", choices=["32", "52", "64"])
    parser.add_argument("--shuffle-cards", type=bool, default=True)
    parser.add_argument("--single-card-mode", type=bool, default=False)
    args = parser.parse_args()

    game_settings = {
        "num_players": args.num_players,
        "deck_size": args.deck_size,
        "shuffle_cards": args.shuffle_cards,
        "single_card_mode": args.single_card_mode,
    }

    # Game/Env laden → Input-Dimension ermitteln
    game = pyspiel.load_game("president", game_settings)
    env = rl_environment.Environment(game)
    input_dim = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # Agent laden
    agent = PPOAgent(input_dim, num_actions)
    if args.model_prefix:
        agent.restore(args.model_prefix)

    # Output-Verzeichnis = utils/agent_plots
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "agent_plots"
    print(f"Speichere nach: {out_dir}")

    # Kurze Übersicht
    print("\nPolicy params:")
    for n, p in agent._policy.named_parameters():
        print(n, tuple(p.shape))
    print("\nValue params:")
    for n, p in agent._value.named_parameters():
        print(n, tuple(p.shape))

    print("\n=== torchinfo ===")
    summary(agent._policy, input_size=(1, input_dim))
    summary(agent._value,  input_size=(1, input_dim))

    # torchview-Render in utils/agent_plots
    render_torchview(agent._policy, "policy", input_dim, out_dir)
    render_torchview(agent._value,  "value",  input_dim, out_dir)


if __name__ == "__main__":
    main()
