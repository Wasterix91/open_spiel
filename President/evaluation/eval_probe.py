# -*- coding: utf-8 -*-
# evaluation/eval_probe.py
# Policy probe from manually provided Info-State vectors (no engine stepping).
# - You provide 4 Info-State strings (one per player) OR a text file with 4 lines
# - You choose which player is to act (current_player)
# - We derive legal actions from (nTop, TopRank, counts) + game action map
# - We ask each configured agent (PPO or heuristic) what they would do (greedy)
# - Outputs nicely printed decisions and CSVs (probs & summary), following CONFIG.

import os, re, json, argparse, numpy as np, pandas as pd, torch, pyspiel
from typing import List, Tuple, Dict

from agents import ppo_agent as ppo
from utils.strategies import STRATS
from utils.load_save_a1_ppo import load_checkpoint_ppo

# =======================
# eval_probe.py — CONFIG
# =======================
CONFIG = {
    "game_settings": {
        "num_players": 4,
        "deck_size": "64",          # supports 12,16,20,24,32,64,52
        "shuffle_cards": True,
        "single_card_mode": False,
    },

    # Player-Setup (wie in eval_micro): ppo/dqn brauchen family/version/episode; Heuristiken via STRATS-Key
    "players": [
        {"name": "P0", "type": "ppo", "family": "k3a1", "version": "05", "episode": 20_000, "from_pid": 0},
        {"name": "P1", "type": "ppo", "family": "k1a1", "version": "57", "episode": 200, "from_pid": 0},
        {"name": "P2", "type": "ppo", "family": "k3a1", "version": "05", "episode": 20_000, "from_pid": 0},
        {"name": "P3", "type": "ppo", "family": "k1a1", "version": "57", "episode": 200, "from_pid": 0},
    ],
    # Probe-Parameter
    "player_to_move": 0,            # wessen Policy wir abfragen
    "legal_actions": None,          # z.B. [0, 11, 12, 14]; None => aus InfoState ableiten
    "softmax_temperature": 1.0,     # 1.0 = neutral; nur für Probendarstellung

    # Information-State Eingaben (Strings oder Listen von Zahlen)
    # Beispiel-Format (64er Deck, 14er IST): "4,1,0,3,1,0,3,2,14,14,14,2,2,12"
    "infostates": {
        0: "4,1,0,3,1,0,3,2,14,14,14,2,2,2",
        1: "2,2,2,2,2,2,2,2,14,14,14,0,0,-1",
        2: "0,0,0,0,0,0,0,0,14,14,14,0,0,-1",
        3: "0,0,0,0,0,0,0,0,14,14,14,0,0,-1",
    },

    "output_root": None,  # None => Standard: "<dieses_verzeichnis>/eval_probe"

    "save_files": {
        "action_probs": True,   # csv/action_probs.csv (eine Zeile pro legaler Action und Agent)
        "probe_result": True,   # probe_result.csv (pro Agent: chosen action + text)
        "player_config": True,  # player_config.csv
        "probe_config": True,   # probe_config.json (deine Eingaben)
    },
}

# ---------- Helpers: deck & ranks ----------
def _int_deck_size(ds: str|int) -> int:
    try: return int(str(ds))
    except: return 64

_RANKS_BY_DECK = {
    12: ["Q","K","A"],
    16: ["J","Q","K","A"],
    20: ["10","J","Q","K","A"],
    24: ["9","10","J","Q","K","A"],
    32: ["7","8","9","10","J","Q","K","A"],
    64: ["7","8","9","10","J","Q","K","A"],
    52: ["2","3","4","5","6","7","8","9","10","J","Q","K","A"],
}

def rank_headers_for_game(game, game_settings) -> List[str]:
    ds = _int_deck_size(game_settings.get("deck_size","64"))
    n = game.information_state_tensor_shape()[0] - 6
    if ds in _RANKS_BY_DECK and len(_RANKS_BY_DECK[ds]) == n:
        return _RANKS_BY_DECK[ds]
    base = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
    return base[:n]

# ---------- Action map from game ----------
def build_action_maps(game) -> Tuple[Dict[Tuple[int,str],int], Dict[int,Tuple[int,str]], int]:
    """
    Returns:
       to_id[(k,rank)] = action_id  for "Play k-of-a-kind of <rank>"
       from_id[action_id] = (k, rank)
       pass_id = action id of "Pass" (or None if not found)
    """
    dummy = game.new_initial_state()
    to_id = {}
    from_id = {}
    pass_id = None
    for aid in range(game.num_distinct_actions()):
        try:
            txt = dummy.action_to_string(0, aid)
        except Exception:
            continue
        if txt == "Pass":
            pass_id = aid
            continue
        m = re.match(r"Play\s+(Single|Pair|Triple|Quad|\d+-of-a-kind)\s+of\s+(.+)$", txt)
        if not m: 
            continue
        kind, rank = m.group(1), m.group(2).strip()
        if kind == "Single": k = 1
        elif kind == "Pair": k = 2
        elif kind == "Triple": k = 3
        elif kind == "Quad": k = 4
        else:
            km = re.match(r"(\d+)-of-a-kind", kind)
            if not km: 
                continue
            k = int(km.group(1))
        to_id[(k, rank)] = aid
        from_id[aid] = (k, rank)
    return to_id, from_id, pass_id

# ---------- PPO loader (greedy) ----------
def _ppo_expected_stem(models_root, family, version, seat_on_disk, episode):
    base_dir = os.path.join(models_root, family, f"model_{version}", "models")
    stem = os.path.join(base_dir, f"{family}_model_{version}_agent_p{seat_on_disk}_ep{int(episode):07d}")
    if not (os.path.exists(stem + "_policy.pt") and os.path.exists(stem + "_value.pt")):
        raise FileNotFoundError(f"Missing PPO weights under {stem}_policy.pt/_value.pt")
    return stem

def load_ppo_agent(game, cfg, models_root):
    info_dim    = game.information_state_tensor_shape()[0]
    num_actions = game.num_distinct_actions()
    num_players = game.num_players()
    stem = _ppo_expected_stem(models_root, cfg["family"], cfg["version"], cfg.get("from_pid", 0), cfg["episode"])

    tried = []
    for seat_id_dim in (num_players, 0):
        try:
            ag = ppo.PPOAgent(info_state_size=info_dim, num_actions=num_actions, seat_id_dim=seat_id_dim, device="cpu")
            load_checkpoint_ppo(ag, os.path.dirname(stem), os.path.basename(stem))
            ag._policy.eval(); ag._value.eval()
            if seat_id_dim == 0:
                print(f"[INFO] PPO loaded without seat-one-hot (in_features={info_dim}).")
            else:
                print(f"[INFO] PPO loaded with seat-one-hot (in_features={info_dim+num_players}).")
            return ag
        except Exception as e:
            tried.append(f"seat_id_dim={seat_id_dim}: {e}")
            continue
    raise RuntimeError("Failed to load PPO with/without seat-one-hot:\n  " + "\n  ".join(tried))

def explain_legal_from_ist(ist, rank_labels):
    n_ranks = len(rank_labels)
    counts = list(map(int, ist[:n_ranks]))
    lastp, nTop, topIdx = map(int, ist[n_ranks+3:n_ranks+6])
    top_name = "-" if topIdx < 0 or topIdx >= n_ranks else rank_labels[topIdx]
    print("\n[Probe · IST-Interpretation]")
    print(" Ränge: " + " ".join(f"{r:>3}" for r in rank_labels))
    print(" Count: " + " ".join(f"{c:>3d}" for c in counts))
    print(f" nTop={nTop}, TopRank={top_name}, LastP={lastp}")
    if nTop <= 0:
        print(" ⇒ Freies Ausspiel (alle 1..4-of-a-kind, sofern genug Karten).")
    else:
        print(f" ⇒ Muss {nTop}-of-a-kind > {top_name} schlagen; Pass ist erlaubt.")

def ppo_logits_from_info(ag, info_vec, seat_id, num_players):
    seat_oh = np.zeros(num_players, dtype=np.float32); seat_oh[seat_id] = 1.0
    x = ag._make_input(np.asarray(info_vec, dtype=np.float32), seat_one_hot=seat_oh)
    with torch.no_grad():
        return ag._policy(x).detach().cpu().numpy()

def masked_softmax(logits, legal, temperature: float = 1.0):
    logits = np.asarray(logits, dtype=np.float32)
    if len(legal) == 0:
        return np.zeros_like(logits, dtype=np.float32)
    if temperature <= 0:
        temperature = 1e-6
    mask = np.full_like(logits, -np.inf, dtype=np.float32)
    mask[list(legal)] = logits[list(legal)]
    m = np.max(mask[list(legal)])
    ex = np.exp((mask - m) / float(temperature))
    ex[~np.isfinite(ex)] = 0.0
    s = ex.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.zeros_like(logits, dtype=np.float32)
        p[list(legal)] = 1.0 / len(legal)
        return p
    return ex / s

# ---------- Legal actions from InfoState ----------
def derive_legal_actions_from_ist(game, info_vec, rank_labels, to_action_id, pass_id) -> List[int]:
    """Legal = Pass (if nTop>0) + plays of size:
       - if nTop==0: any k∈{1,2,3,4} with count[r]>=k
       - else: exactly k=nTop and rank strictly higher than TopRankIndex with count[r]>=k
    """
    info_vec = np.asarray(info_vec, dtype=np.float32)
    n_ranks = len(rank_labels)
    counts = list(map(int, info_vec[:n_ranks]))
    p1, p2, p3 = map(int, info_vec[n_ranks:n_ranks+3])
    lastp, nTop, topIdx = map(int, info_vec[n_ranks+3:n_ranks+6])
    legal = []

    def maybe_add(k, r_idx):
        rank = rank_labels[r_idx]
        aid = to_action_id.get((k, rank))
        if aid is not None:
            legal.append(aid)

    if nTop <= 0:
        for r in range(n_ranks):
            c = counts[r]
            if c >= 1: maybe_add(1, r)
            if c >= 2: maybe_add(2, r)
            if c >= 3: maybe_add(3, r)
            if c >= 4: maybe_add(4, r)
    else:
        if pass_id is not None:
            legal.append(pass_id)
        for r in range(n_ranks):
            if r <= topIdx:
                continue
            if counts[r] >= nTop:
                maybe_add(nTop, r)

    return sorted(set(legal))

# ---------- Parse IST ----------
def parse_ist_line(line: str) -> List[float]:
    s = line.strip()
    if s.startswith("[") and s.endswith("]"):
        arr = json.loads(s)
        return [float(x) for x in arr]
    parts = [p.strip() for p in s.split(",")]
    return [float(p) for p in parts if p != ""]

# ---------- Heuristic pick using action semantics ----------
def pick_heuristic_greedy(legal: List[int], from_action_id: Dict[int,Tuple[int,str]], rank_labels: List[str], pass_id: int|None):
    if not legal:
        return None
    # Prefer non-pass actions; rank by (k desc, rank_index desc)
    non_pass = [a for a in legal if a != pass_id]
    if not non_pass:
        return pass_id if pass_id in legal else legal[0]
    rank_pos = {r:i for i,r in enumerate(rank_labels)}
    def key(aid):
        k, r = from_action_id.get(aid, (0, rank_labels[0]))
        return (k, rank_pos.get(r, -1))
    return max(non_pass, key=key)

# =======================
# Main
# =======================
def main():
    ap = argparse.ArgumentParser(description="Probe agent decisions from Info-State (no engine).")
    ap.add_argument("--info", nargs=4, metavar=("P0","P1","P2","P3"),
                    help='Four IST lines like "1,0,3,...,-1" (one per player). If omitted, take CONFIG[infostates].')
    ap.add_argument("--info_file", type=str, default=None,
                    help="Text file with exactly 4 lines (P0..P3) containing IST lines.")
    ap.add_argument("--current_player", type=int, default=None, help="Which player to act (0..3). Default from CONFIG.")
    ap.add_argument("--models_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    # Resolve settings from CONFIG (+ optional CLI overrides)
    game_settings = CONFIG["game_settings"]
    players_cfg   = CONFIG["players"]
    current_player = args.current_player if args.current_player is not None else int(CONFIG.get("player_to_move", 0))
    models_root = args.models_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    out_root = args.out_dir or CONFIG.get("output_root") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_probe")
    os.makedirs(out_root, exist_ok=True)
    existing = sorted([d for d in os.listdir(out_root) if d.startswith("probe_")])
    next_id = int(existing[-1].split("_")[-1]) + 1 if existing else 1
    run_dir = os.path.join(out_root, f"probe_{next_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # Load/parse ISTs
    if args.info_file:
        with open(args.info_file, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]
        if len(lines) != 4:
            raise SystemExit("info_file must contain exactly 4 non-empty lines (P0..P3).")
        infos = [parse_ist_line(ln) for ln in lines]
    elif args.info is not None:
        if len(args.info) != 4:
            raise SystemExit("--info needs exactly 4 items.")
        infos = [parse_ist_line(ln) for ln in args.info]
    else:
        # from CONFIG
        infos = [parse_ist_line(CONFIG["infostates"][i]) for i in range(4)]

    # Build game + maps
    game = pyspiel.load_game("president", game_settings)
    rank_labels = rank_headers_for_game(game, game_settings)
    to_action_id, from_action_id, pass_id = build_action_maps(game)
    NUM_PLAYERS = game.num_players()
    NUM_ACTIONS = game.num_distinct_actions()

    pid = int(current_player)
    if not (0 <= pid < NUM_PLAYERS):
        raise SystemExit("current_player must be 0..3.")

    ist = infos[pid]
    n_ranks = len(rank_labels)
    expected_len = n_ranks + 6
    if len(ist) != expected_len:
        raise SystemExit(f"InfoState length mismatch: got {len(ist)}, expected {expected_len} for deck_size={game_settings['deck_size']}.")

    # Legal actions: from CONFIG override or derive from IST
    legal_cfg = CONFIG.get("legal_actions")
    if legal_cfg is not None:
        legal = sorted(set(int(a) for a in legal_cfg))
        print("[INFO] Using CONFIG['legal_actions'] override:", legal)
    else:
        legal = derive_legal_actions_from_ist(game, ist, rank_labels, to_action_id, pass_id)
        explain_legal_from_ist(ist, rank_labels)  # nur erklären, wenn wir sie hergeleitet haben


    # Prepare outputs
    probs_csv = os.path.join(run_dir, "action_probs.csv")
    result_csv = os.path.join(run_dir, "probe_result.csv")
    player_cfg_csv = os.path.join(run_dir, "player_config.csv")
    probe_cfg_json = os.path.join(run_dir, "probe_config.json")

    if CONFIG["save_files"].get("player_config", True):
        pd.DataFrame(players_cfg).to_csv(player_cfg_csv, index=False)
    if CONFIG["save_files"].get("probe_config", True):
        with open(probe_cfg_json, "w") as f:
            json.dump({
                "game_settings": game_settings,
                "players": players_cfg,
                "player_to_move": pid,
                "legal_actions": legal if legal_cfg is not None else "derived",
                "infostates": {i: infos[i] for i in range(4)},
                "rank_labels": rank_labels,
                "models_root": models_root,
                "temperature": CONFIG.get("softmax_temperature", 1.0),
            }, f, indent=2)

    rows_probs = []
    rows_result = []

    dummy = game.new_initial_state()
    print(f"\n== Probe (deck={game_settings['deck_size']}, current_player=P{pid}) ==")
    print("Legal actions:")
    for aid in legal:
        print(f"  {aid:3d}: {dummy.action_to_string(pid, aid)}")

    # Iterate agents
    for i, cfg in enumerate(players_cfg):
        name = cfg.get("name", f"Agent{i}")
        kind = cfg.get("type")

        if kind in STRATS:
            chosen = pick_heuristic_greedy(legal, from_action_id, rank_labels, pass_id)
            if chosen is None:
                continue
            # uniform probs for display
            probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
            if legal:
                probs[legal] = 1.0 / len(legal)
            rows_result.append({"agent": name, "type": kind, "chosen_id": int(chosen),
                                "chosen_text": dummy.action_to_string(pid, chosen)})
            for a in legal:
                rows_probs.append({"agent": name, "type": kind, "action_id": a,
                                   "action_text": dummy.action_to_string(pid, a),
                                   "prob": float(probs[a]), "score": float(probs[a]), "chosen": int(a==chosen)})
            print(f"- {name:20s} -> {dummy.action_to_string(pid, chosen)}")
            continue

        if kind == "ppo":
            try:
                ag = load_ppo_agent(game, cfg, models_root)
            except Exception as e:
                print(f"[WARN] skip {name}: cannot load PPO ({e})")
                continue
            logits = ppo_logits_from_info(ag, ist, pid, NUM_PLAYERS)
            probs = masked_softmax(logits, legal, temperature=CONFIG.get("softmax_temperature", 1.0))
            chosen = int(np.argmax(probs))
            rows_result.append({"agent": name, "type": "ppo", "chosen_id": chosen,
                                "chosen_text": dummy.action_to_string(pid, chosen)})
            for a in legal:
                rows_probs.append({"agent": name, "type": "ppo", "action_id": a,
                                   "action_text": dummy.action_to_string(pid, a),
                                   "prob": float(probs[a]), "score": float(logits[a]), "chosen": int(a==chosen)})
            print(f"- {name:20s} -> {dummy.action_to_string(pid, chosen)}")
            continue

        # DQN not supported without true engine/obs
        if kind == "dqn":
            print(f"[INFO] DQN '{name}' not supported in probe mode (needs observation & engine). Skipping.")
            continue

        print(f"[WARN] Unknown agent type {kind} for {name} (skipped).")

    # Save CSVs
    if rows_probs and CONFIG["save_files"].get("action_probs", True):
        pd.DataFrame(rows_probs, columns=["agent","type","action_id","action_text","prob","score","chosen"]).to_csv(probs_csv, index=False)
        print(f"📄 action_probs.csv → {probs_csv}")
    if rows_result and CONFIG["save_files"].get("probe_result", True):
        pd.DataFrame(rows_result, columns=["agent","type","chosen_id","chosen_text"]).to_csv(result_csv, index=False)
        print(f"📄 probe_result.csv → {result_csv}")

    if not rows_probs and not rows_result:
        print("\n[WARN] No output rows generated (no agents or load failures).")

if __name__ == "__main__":
    main()
