# TODO

## Geplante Funktionen

- [ ] **Interaktive Demo:** [`President/evaluation/play_human.py`](President/evaluation/play_human.py) als einfach nutzbaren Einstieg ausbauen, damit Menschen gegen vorhandene Heuristiken sowie trainierte PPO- und DQN-Agents spielen können.
- [ ] **Agentenvergleich:** Einen kompakten Evaluationsaufruf mit frei wählbaren Agenten und Gegnerkonstellationen bereitstellen.
- [ ] **Beispieltraining:** Ein kleines, nachvollziehbares PPO- oder DQN-Training mit wenigen Episoden dokumentieren.
- [ ] **Optionale Oberfläche:** Eine benutzerfreundliche Oberfläche für Partien gegen Agents oder Freunde prüfen.

## Bekannte technische Auffälligkeiten

- [ ] **C++-Tests:** Die Tests unter [`open_spiel/games/president/`](open_spiel/games/president/) von übernommenen Poker-Tests bereinigen und um gezielte President-Invarianten ergänzen.
- [ ] **Utility-Klassifikation:** Prüfen und dokumentieren, ob `GameType::Utility::kZeroSum` durch `kConstantSum` ersetzt werden sollte, da die platzierungsabhängigen Returns konstant, aber nicht zu null summieren.
