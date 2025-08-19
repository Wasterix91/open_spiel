"""
Heuristische Strategien für OpenSpiel 'president'.
Alle Funktionen: f(state) -> int (Action-ID).
"""

from __future__ import annotations
import numpy as np

# ---------- Helfer ----------

def _decode_actions(state, include_pass=False):
    """Gibt [(action_id, text), ...] für den aktuellen Spieler zurück."""
    pid = state.current_player()
    decoded = [(a, state.action_to_string(pid, a)) for a in state.legal_actions()]
    if not include_pass:
        decoded = [(a, s) for (a, s) in decoded if a != 0]  # 0 = Pass
    return decoded

def _combo_size_priority(text: str) -> int:
    # Einheitliche Priorität: 1..4 und ggf. 5..8-of-a-kind
    if "Quad"   in text: return 4
    if "Triple" in text: return 3
    if "Pair"   in text: return 2
    if "Single" in text: return 1
    # Deckt 64er-Deck ab (5..8-of-a-kind)
    for k in range(8, 4, -1):
        if f"{k}-of-a-kind" in text:
            return k
    return 0

def _rank_priority(text: str) -> int:
    # Für aggressive/smart – extrahiert Rang (7..A). Unbekannt -> -1.
    # Text sieht z.B. aus wie: "Play Single 9" / "Play Pair K"
    try:
        rank = text.split()[-1]
    except Exception:
        return -1
    order = {"7":0, "8":1, "9":2, "10":3, "J":4, "Q":5, "K":6, "A":7}
    return order.get(rank, -1)

# ---------- Strategien ----------

def random_action(state) -> int:
    """Völlig zufällig (inkl. Pass)."""
    return int(np.random.choice(state.legal_actions()))

def random2(state) -> int:
    """Zufällig, aber vermeidet Pass wenn eine andere Aktion möglich ist."""
    legal = list(state.legal_actions())
    if len(legal) > 1 and 0 in legal:
        legal = [a for a in legal if a != 0]
    return int(np.random.choice(legal))

def single_only(state) -> int:
    """Spielt, falls möglich, irgendein Single; sonst Pass."""
    dec = _decode_actions(state)
    if not dec:
        return 0
    singles = [x for x in dec if "Single" in x[1]]
    return int(singles[0][0]) if singles else 0

def max_combo(state) -> int:
    """Wählt die größtmögliche Gruppengröße (Tie-Break: höhere Action-ID)."""
    dec = _decode_actions(state)
    if not dec:
        return 0
    return int(max(dec, key=lambda x: (_combo_size_priority(x[1]), -x[0]))[0])

def max_combo2(state) -> int:
    """Wie max_combo, aber berücksichtigt 5..8-of-a-kind explizit (64er-Deck)."""
    dec = _decode_actions(state)
    if not dec:
        return 0
    return int(max(dec, key=lambda x: (_combo_size_priority(x[1]), -x[0]))[0])

def aggressive(state) -> int:
    """Bevorzugt die höchste Rangstufe (innerhalb der legalen Züge)."""
    dec = _decode_actions(state)
    if not dec:
        return 0
    return int(max(dec, key=lambda x: _rank_priority(x[1]))[0])

def smart(state) -> int:
    """Erst größte Kombogröße, innerhalb dieser den niedrigsten Rang (sparsam)."""
    dec = _decode_actions(state)
    if not dec:
        return 0
    groups = {}
    for a, s in dec:
        groups.setdefault(_combo_size_priority(s), []).append((a, s))
    for size in sorted(groups.keys(), reverse=True):
        cand = groups[size]
        a, _ = min(cand, key=lambda x: _rank_priority(x[1]))
        return int(a)
    return 0

# ---------- Export-Mapping ----------

STRATS = {
    "random":       random_action,
    "random2":      random2,
    "single_only":  single_only,
    "max_combo":    max_combo,
    "max_combo2":   max_combo2,
    "aggressive":   aggressive,
    "smart":        smart,
}

def get_strategy(name_or_callable):
    """Gibt eine Strategie-Funktion zurück (oder validiert bereits übergebene Callables)."""
    if callable(name_or_callable):
        return name_or_callable
    try:
        return STRATS[name_or_callable]
    except KeyError as e:
        raise ValueError(f"Unbekannte Strategie: {name_or_callable!r}. "
                         f"Verfügbar: {sorted(STRATS.keys())}") from e
