# -*- coding: utf-8 -*-
# Interactive CLI: Du spielst Sitz 0 (Human). Gegner = Heuristiken oder geladene Policies.
# Jetzt mit Logging wie eval_micro:
#  - Run-Ordner nach erfolgreichem Laden
#  - game_log.csv, action_probs.csv, initial_info_state_tensor.csv, player_config.csv, game_settings.json, summary.json
#  - Greedy-Eval für Policies (PPO/DQN)
#  - Abbruch, wenn explizite Checkpoints fehlen

import os, json, numpy as np, pandas as pd, torch, pyspiel

from agents import ppo_agent as ppo
from agents import dqn_agent as dqn
from agents import v_table_agent
from utils.strategies import STRATS
from collections import namedtuple

from utils.load_save_a1_ppo import load_checkpoint_ppo
from utils.load_save_a2_dqn import load_checkpoint_dqn

# ===== Ausgabe für InfoState steuern =====
SHOW_INFOSTATE = True     # auf False setzen, wenn du die Ausgabe mal nicht willst
INFOSTATE_DECIMALS = 3    # Rundung der Werte
INFOSTATE_PER_LINE = 16   # wie viele Werte pro Zeile drucken

# ===== Spiel-Setup =====
GAME_SETTINGS = {
    "num_players": 4,
    "deck_size": "16",
    "shuffle_cards": True,
    "single_card_mode": False,
}

# Gegner: Heuristiken oder Policies; für Policies sind family/version/episode Pflicht!
# from_pid: von welchem Seat die Gewichte auf der Platte geladen werden (für shared policy etc.)
""" OPPONENTS = [
    {"name": "P1", "type": "dqn", "family": "k1a2", "version": "38", "episode": 75_000, "from_pid": 0},
    {"name": "P2", "type": "dqn", "family": "k1a2", "version": "38", "episode": 75_000, "from_pid": 0},
    # Beispiel PPO:
    {"name": "P3", "type": "dqn", "family": "k1a2", "version": "38", "episode": 75_000, "from_pid": 0},
    #{"name": "Opp3", "type": "ppo", "family": "k1a1", "version": "55", "episode": 80, "from_pid": 0},  # Player3 für 12 Karten
] """

OPPONENTS = [
    {"name": "dqn", "type": "dqn", "family": "k1a2", "version": "46", "episode": 40_000, "from_pid": 0},
    {"name": "v_table", "type": "v_table"},
    {"name": "max_combo", "type": "max_combo"},
]

# ===== Pfade & kleine Utils =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def _fatal(msg, tried=None):
    lines = [f"[FATAL] {msg}"]
    if tried:
        lines.append("  Versucht:")
        for t in tried[:30]:
            lines.append(f"    {t}")
        if len(tried) > 30:
            lines.append("    ...")
    raise SystemExit("\n".join(lines))

def _norm_episode(ep):
    s = str(ep).replace("_","").strip()
    if s.isdigit():
        return int(s)
    _fatal(f"Episode muss explizit gesetzt sein (int oder numerischer String), erhalten: {ep!r}")

def _ppo_expected_stem(family: str, version: str, seat_on_disk: int, episode: int):
    base_dir = os.path.join(MODELS_ROOT, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{episode:07d}")
    pol, val = stem + "_policy.pt", stem + "_value.pt"
    if not (os.path.exists(pol) and os.path.exists(val)):
        _fatal(
            f"PPO-Checkpoint fehlt: family={family}, version={version}, seat={seat_on_disk}, episode={episode}",
            tried=[pol, val],
        )
    return stem

def _dqn_expected_stem(family: str, version: str, seat_on_disk: int, episode: int):
    base_dir = os.path.join(MODELS_ROOT, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{episode:07d}")
    # neue Trainings-Namen
    qnet = stem + "_qnet.pt"
    tgt  = stem + "_tgt.pt"
    # legacy-Variante (falls vorhanden)
    legacy_q = stem + "_q.pt"

    # Mit neuem Loader reicht _qnet.pt (Target optional) ODER Legacy _q.pt
    if not (os.path.exists(qnet) or os.path.exists(legacy_q)):
        _fatal(
            f"DQN-Checkpoint fehlt: family={family}, version={version}, seat={seat_on_disk}, episode={episode}",
            tried=[qnet, tgt, legacy_q],
        )
    return stem


def _alias_dqn_attrs(agent):
    if not hasattr(agent, "q_net") and hasattr(agent, "q_network"):
        agent.q_net = agent.q_network
    if not hasattr(agent, "target_net") and hasattr(agent, "target_network"):
        agent.target_net = agent.target_network
    return agent

# ===== Softmax mit Legal-Masking (NumPy) =====
def _masked_softmax_numpy(scores, legal):
    scores = np.asarray(scores, dtype=np.float32)
    if len(legal) == 0:
        # keine legalen Aktionen → gebe 0-Vector zurück (keine Verteilung)
        return np.zeros_like(scores, dtype=np.float32)
    mask = np.full_like(scores, -np.inf, dtype=np.float32)
    mask[list(legal)] = scores[list(legal)]
    m = np.max(mask[list(legal)])
    ex = np.exp(mask - m)
    ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.zeros_like(scores, dtype=np.float32)
        p[list(legal)] = 1.0 / len(legal)
        return p
    return ex / s


# ===== Loader =====
def _load_ppo_agent(info_dim, num_actions, seat_id_dim, *, family, version, episode, from_pid, device="cpu"):
    ep = _norm_episode(episode)
    stem = _ppo_expected_stem(family, version, from_pid, ep)
    ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim, device=device)
    weights_dir, tag = os.path.dirname(stem), os.path.basename(stem)
    try:
        load_checkpoint_ppo(ag, weights_dir, tag)
        ag._policy.eval(); ag._value.eval()
        return ag
    except Exception as e:
        _fatal("Fehler beim Laden via load_checkpoint_ppo(...).", tried=[f"{stem}: {e}"])

def _load_dqn_agent(obs_dim, num_actions, *, family, version, episode, from_pid, num_players, device="cpu"):
    ep = _norm_episode(episode)
    stem = _dqn_expected_stem(family, version, from_pid, ep)
    tried = []
    for state_size in (obs_dim, obs_dim + num_players):  # ohne/mit Seat-One-Hot
        ag = dqn.DQNAgent(state_size=state_size, num_actions=num_actions, device=device)
        _alias_dqn_attrs(ag)
        weights_dir, tag = os.path.dirname(stem), os.path.basename(stem)
        try:
            load_checkpoint_dqn(ag, weights_dir, tag)
            if hasattr(ag, "epsilon"): ag.epsilon = 0.0
            return ag
        except Exception as e:
            tried.append(f"{stem} (state_size={state_size}): {e}")
    _fatal("Fehler beim Laden von DQN-Gewichten.", tried=tried)

def load_opponent(cfg, pid, game):
    info_dim    = game.information_state_tensor_shape()[0]
    obs_dim     = game.observation_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    NUM_PLAYERS = game.num_players()

    kind = cfg.get("type")

    if kind == "ppo":
        if not all(k in cfg for k in ("family","version","episode")):
            _fatal(f"PPO-Gegner P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
        # zuerst MIT, dann OHNE Seat-One-Hot versuchen
        try:
            return _load_ppo_agent(info_dim, num_actions, NUM_PLAYERS,
                                   family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                   from_pid=cfg.get("from_pid", pid))
        except SystemExit:
            return _load_ppo_agent(info_dim, num_actions, 0,
                                   family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                                   from_pid=cfg.get("from_pid", pid))

    if kind == "dqn":
        if not all(k in cfg for k in ("family","version","episode")):
            _fatal(f"DQN-Gegner P{pid}: 'family', 'version' und 'episode' sind Pflicht. Erhalten: {cfg}")
        return _load_dqn_agent(obs_dim, num_actions,
                               family=cfg["family"], version=cfg["version"], episode=cfg["episode"],
                               from_pid=cfg.get("from_pid", pid), num_players=NUM_PLAYERS)

    if kind == "v_table":
        return v_table_agent.ValueTableAgent("agents/tables/v_table_4_4_4")

    if kind in STRATS:
        return STRATS[kind]

    _fatal(f"Unbekannter Gegner-Typ: {kind}")

# ===== Vorwärtswege (Logits/Qs) =====
def _ppo_logits(agent: ppo.PPOAgent, info_state_1d: np.ndarray, seat_id: int, num_players: int):
    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[seat_id] = 1.0
    x = agent._make_input(info_state_1d, seat_one_hot=seat_oh)
    with torch.no_grad():
        return agent._policy(x).detach().cpu().numpy()

def _dqn_qvalues(agent: dqn.DQNAgent, obs_1d: np.ndarray, seat_id: int, num_players: int):
    x = np.asarray(obs_1d, dtype=np.float32)
    try:
        in_features = agent.q_network.net[0].in_features
    except Exception:
        in_features = x.shape[0]
    if x.shape[0] < in_features:
        extra = in_features - x.shape[0]
        if extra == num_players:
            seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[seat_id] = 1.0
            x = np.concatenate([x, seat_oh], axis=0)
        else:
            x = np.concatenate([x, np.zeros(extra, dtype=np.float32)], axis=0)
    elif x.shape[0] > in_features:
        x = x[:in_features]
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32, device=getattr(agent, "device", "cpu")).unsqueeze(0)
        q = agent.q_network(xt).squeeze(0).detach().cpu().numpy()
    return q

AgentOut = namedtuple("AgentOut", ["action"])

def choose_by_agent_with_probs(agent, state, pid, NUM_PLAYERS):
    legal = state.legal_actions(pid)

    # Heuristik (callable)
    if callable(agent):
        a = int(agent(state))
        if a not in legal: a = int(np.random.choice(legal))
        probs = np.zeros(state.num_distinct_actions(), dtype=np.float32)
        probs[legal] = 1.0 / max(1, len(legal))
        scores = probs.copy()
        # greedy (Uniform → egal, wir markieren chosen separat)
        return AgentOut(a), probs, scores

    # PPO: greedy über masked softmax(Logits)
    if isinstance(agent, ppo.PPOAgent):
        info_vec = np.array(state.information_state_tensor(pid), dtype=np.float32)
        logits = _ppo_logits(agent, info_vec, pid, NUM_PLAYERS)
        probs  = _masked_softmax_numpy(logits, legal)
        action = int(np.argmax(probs))
        return AgentOut(action), probs, logits

    # DQN: greedy über Qs (Pseudo-Probs = Softmax über legalen Qs)
    if isinstance(agent, dqn.DQNAgent):
        obs_vec = np.array(state.observation_tensor(pid), dtype=np.float32)
        qvals = _dqn_qvalues(agent, obs_vec, pid, NUM_PLAYERS)
        probs = _masked_softmax_numpy(qvals, legal)
        action = int(np.argmax(probs))
        return AgentOut(action), probs, qvals

    _fatal("Unbekannter Agententyp in choose_by_agent_with_probs.")

# ---- hübsche Ausgabe des InfoState (deck-size aware) ----

def _int_deck_size():
    try:
        return int(str(GAME_SETTINGS.get("deck_size", "64")))
    except Exception:
        return 64

_RANKS_BY_DECK = {
    12: ["Q","K","A"],
    16: ["J","Q","K","A"],
    20: ["10","J","Q","K","A"],
    24: ["9","10","J","Q","K","A"],
    32: ["7","8","9","10","J","Q","K","A"],
    64: ["7","8","9","10","J","Q","K","A"],
    52: ["2","3","4","5","6","7","8","9","10","J","Q","K","A"],
}

def _rank_headers_for_current_game(n_ranks: int) -> list[str]:
    """Wähle die korrekten Rang-Labels basierend auf Deckgröße.
    Index 0 ist die niedrigste Karte im verwendeten Deck."""
    ds = _int_deck_size()
    if ds in _RANKS_BY_DECK and len(_RANKS_BY_DECK[ds]) == n_ranks:
        return _RANKS_BY_DECK[ds]
    base = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
    return base[:n_ranks]

def print_info_state(state, pid, decimals=None, per_line=None):
    """
    Kompakte, deck-size-abhängige Ausgabe:
    <R1 R2 ...> |  P1  P2  P3 | LastP nTop TopRank
    Werte: counts, draw sizes, letzte(r) Spieler/Top-Stack/oberste Karte (als Rang).
    """
    vec = np.asarray(state.information_state_tensor(pid), dtype=np.float32)
    L = vec.shape[0]

    # Erwartetes Layout: [0..n-1]=Rang-Counts, [n..n+2]=P1,P2,P3, [n+3..n+5]=LastP,nTop,TopRank
    if L >= 10:
        n_ranks = L - 10
        r = list(map(int, vec[:n_ranks]))
        try:
            p1, p2, p3 = map(int, vec[n_ranks:n_ranks+3])
            lastp, ntop, topidx = map(int, vec[n_ranks+3:n_ranks+6])
        except Exception:
            # Fallback auf Rohdump, falls das Layout abweicht
            print(f"[InfoState P{pid}] len={L}")
            print("  " + " ".join(str(int(x)) if float(x).is_integer() else f"{x:.3f}" for x in vec))
            return

        headers = _rank_headers_for_current_game(n_ranks)
        top_name = headers[topidx] if 0 <= topidx < n_ranks else "-"

        colw = max(2, max(len(h) for h in headers))
        hdr_l = " ".join(f"{h:>{colw}}" for h in headers)
        hdr_r = " ".join(f"{h:>3}" for h in ("P1","P2","P3"))
        print(f"[InfoState P{pid}]")
        print(f"{hdr_l} | {hdr_r} | {'LastP':>5} {'nTop':>4} {'TopRank':>7}")

        left  = " ".join(f"{v:>{colw}d}" for v in r)
        right = " ".join(f"{v:>3d}" for v in (p1, p2, p3))
        print(f"{left} | {right} | {lastp:>5d} {ntop:>4d} {top_name:>7}")
        return

    # Fallback für unbekanntes Layout
    print(f"[InfoState P{pid}] len={L}")
    print("  " + " ".join(str(int(x)) if float(x).is_integer() else f"{x:.3f}" for x in vec))



# ===== Initialer InfoState-Tensor dump =====
def write_initial_ist_csv(game, state, csv_dir, filename="initial_info_state_tensor.csv"):
    info_dim = game.information_state_tensor_shape()[0]
    rows = []
    for pid in range(game.num_players()):
        ist = state.information_state_tensor(pid)
        vec = list(map(float, ist))[:info_dim]
        row = {"player": pid}
        row.update({f"f{i}": vec[i] for i in range(info_dim)})
        rows.append(row)
    cols = ["player"] + [f"f{i}" for i in range(info_dim)]
    out = os.path.join(csv_dir, filename)
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    print(f"Initialer InformationStateTensor gespeichert: {out}")

# ===== Main =====
def main():
    game = pyspiel.load_game("president", GAME_SETTINGS)
    NUM_PLAYERS = game.num_players()

    # Gegner laden (bricht ggf. mit FATAL ab)
    opponents = [load_opponent(OPPONENTS[i-1], i, game) for i in [1,2,3]]

    # Run-Verzeichnisse (erst nach erfolgreichem Laden erstellen)
    eval_root = os.path.join(BASE_DIR, "eval_micro")
    existing = sorted([d for d in os.listdir(eval_root)] if os.path.isdir(eval_root) else [])
    existing = [d for d in existing if d.startswith("eval_micro_")]
    next_run_num = int(existing[-1].split("_")[-1]) + 1 if existing else 1

    run_dir   = os.path.join(eval_root, f"eval_micro_{next_run_num:02d}")
    csv_dir   = os.path.join(run_dir, "csv")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    CONFIG_CSV = os.path.join(run_dir, "config.csv")
    LOG_CSV    = os.path.join(csv_dir, "game_log.csv")
    PROBS_CSV  = os.path.join(csv_dir, "action_probs.csv")

    # Dateien wie in eval_micro
    with open(os.path.join(run_dir, "game_settings.json"), "w") as f:
        json.dump(GAME_SETTINGS, f, indent=2)

    player_config = [{"name":"P0","type":"human"}] + OPPONENTS
    pd.DataFrame(player_config).to_csv(os.path.join(run_dir, "player_config.csv"), index=False)
    print(f"Eval-Micro Run-Ordner (interactive): {run_dir}")

    pd.DataFrame([{
        "game_settings": json.dumps(GAME_SETTINGS),
        "players": json.dumps(player_config),
    }]).to_csv(CONFIG_CSV, index=False)

    # Spiel starten
    rows_log = []
    rows_probs = []
    state = game.new_initial_state()

    # Initialer IST
    write_initial_ist_csv(game, state, csv_dir)
    turn = 0

    print("\n== Human Play: Du bist Player 0 ==")
    print("Mit Enter bestätigst du die Zahl neben der gewünschten Action-ID.\n")

    while not state.is_terminal():
        pid = state.current_player()
        legal = state.legal_actions(pid)
        legal_txt = [state.action_to_string(pid, a) for a in legal]

        if pid == 0:
            if SHOW_INFOSTATE:
                print()  # Leerzeile vor dem InfoState
                print_info_state(state, 0)

            print("\n--- Dein Zug ---")
            for a in legal:
                print(f"{a:3d}: {state.action_to_string(0, a)}")
            choice = None
            while choice not in legal:
                try:
                    text = input("Action-ID wählen: ").strip()
                    choice = int(text)
                except Exception:
                    choice = None
            action = int(choice)

            # Für PROBS-CSV füllen wir (wie eval_micro bei Heuristik) uniform über legal
            probs = np.zeros(game.num_distinct_actions(), dtype=np.float32)
            if len(legal) > 0:
                probs[legal] = 1.0 / len(legal)
            scores = probs.copy()
        else:
            ao, probs, scores = choose_by_agent_with_probs(opponents[pid-1], state, pid, NUM_PLAYERS)
            action = int(ao.action)

        # Haupt-Log
        rows_log.append({
            "turn": turn, "player": pid,
            "legal_actions": str(list(zip(legal, legal_txt))),
            "chosen_action": f"{action} ({state.action_to_string(pid, action)})",
        })

        # Probs-Log (nur legale Aktionen)
        for a, txt in zip(legal, legal_txt):
            rows_probs.append({
                "turn": turn,
                "player": pid,
                "action_id": a,
                "action_text": txt,
                "prob": float(probs[a]),  # bei P0 ist probs bereits uniform über legal
                "score": float(scores[a]),
                "chosen": int(a == action),
            })

        print(f"P{pid} spielt: {state.action_to_string(pid, action)}")
        if pid == 0:
            print()
        state.apply_action(action)
        turn += 1

    # Ende & Dateien schreiben
    rets = state.returns()
    rows_log.append({"turn": turn, "player": "terminal", "legal_actions": "", "chosen_action": ""})

    pd.DataFrame(rows_log).to_csv(LOG_CSV, index=False)
    pd.DataFrame(rows_probs, columns=["turn","player","action_id","action_text","prob","score","chosen"]).to_csv(PROBS_CSV, index=False)
    print(f"Game-Log gespeichert: {LOG_CSV}")
    print(f"Aktions-Wahrscheinlichkeiten gespeichert: {PROBS_CSV}")

    summary = {"micro_id": next_run_num, "num_turns": turn, "returns": rets}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n== Spielende ==")
    print("Returns:", rets)
    best = int(np.argmax(rets))
    print(f"Sieger: Player {best}")

if __name__ == "__main__":
    main()
