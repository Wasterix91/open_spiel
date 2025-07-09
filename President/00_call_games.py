import pyspiel

# Liste aller registrierten Spiele anzeigen
all_games = pyspiel.registered_names()

# Ausgabe
for game in all_games:
    print(game)