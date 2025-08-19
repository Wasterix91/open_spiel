# utils/deck.py
DECK_TO_RANKS = {12: 3, 16: 4, 20: 5, 24: 6, 32: 8, 52: 13, 64: 8}

def ranks_for_deck(total_cards: int) -> int:
    # KeyError ist gewollt, falls eine ungültige Deckgröße verwendet wird.
    return DECK_TO_RANKS[total_cards]
