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
    """Wählt die größtmögliche Gruppengröße (Tie-Break: niedrigere Action-ID)."""
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

def chatgpt(state) -> int:
    """
    Starke, situationsabhängige Heuristik:
    - Neuer Stich: größte Kombogröße, innerhalb dessen niedrigster Rang.
    - Mitgehen: sparsamer Gewinn (niedrigster Rang), moderate Präferenz für größere Kombos.
                Passt, wenn nur sehr hohe Ränge (K/A) nötig wären.
    """
    # Alle legalen Züge (inkl. Pass) dekodieren
    pid = state.current_player()
    legal = list(state.legal_actions())
    include_pass = 0 in legal
    decoded_all = [(a, state.action_to_string(pid, a)) for a in legal]

    # Für Bewertung: nur Nicht-Pass
    decoded = [(a, s) for (a, s) in decoded_all if a != 0]
    if not decoded:
        # Entweder nur Pass oder gar nichts
        return 0 if include_pass else int(np.random.choice(legal))

    # Hilfsfunktionen nutzen
    def combo_size(s: str) -> int:
        c = _combo_size_priority(s)
        return c if c > 0 else 1

    def rank_prio(s: str) -> int:
        return _rank_priority(s)

    # Unterscheiden: Neuer Stich vs. Mitgehen
    new_trick = not include_pass

    # Basispunkte berechnen
    scored = []
    max_rank_val = 7  # bei 32er/64er Deck: A -> 7 (siehe _rank_priority)

    if new_trick:
        # Groß rauslegen, aber hohe Ränge sparen
        for a, s in decoded:
            cs = combo_size(s)
            rp = rank_prio(s)
            # Score: große Kombos stark bevorzugt, innerhalb dessen niedriger Rang
            score = 1000 * cs - 10 * rp
            # Kleine 'Sparprämie' für sehr niedrige Ränge
            if rp <= 2:
                score += 3
            scored.append((score, -a, a))
    else:
        # Mitgehen: sparsam gewinnen, größere Kombos leicht bevorzugt
        # Falls nur sehr hohe Ränge verfügbar wären, lieber passen
        min_rp = min(rank_prio(s) for (_, s) in decoded)
        max_rp = max(rank_prio(s) for (_, s) in decoded)
        only_very_high = min_rp >= 6  # nur K/A schlagen den Stich

        if include_pass and only_very_high:
            return 0  # Highcards fürs Endgame aufsparen

        for a, s in decoded:
            cs = combo_size(s)
            rp = rank_prio(s)
            # Grundscore: größere Kombos nützen (Kartenabbau), aber sparsam (niedriger Rang besser)
            score = 100 * cs - 2 * rp
            # Bonus, wenn wir Top-Rang spielen (schließt oft den Stich)
            if rp == max_rank_val:
                score += 15
            # Singles leicht benachteiligen, wenn auch Mehrlinge möglich sind
            if cs == 1 and any(combo_size(s2) > 1 for (_, s2) in decoded):
                score -= 8
            scored.append((score, -a, a))

    # Beste Aktion wählen (mit stabiler Tie-Breaker-Reihenfolge via -a)
    scored.sort(reverse=True)
    return int(scored[0][2])


# ---------- Export-Mapping aktualisieren ----------

STRATS = {
    "random":       random_action,
    "random2":      random2,
    "single_only":  single_only,
    "max_combo":    max_combo,
    "aggressive":   aggressive,
    "smart":        chatgpt,      
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
