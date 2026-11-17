# Reinforcement Learning für kompetitive Kartenspiele

Dieses Repository enthält die Implementierung zu meiner Masterarbeit **„Reinforcement Learning für kompetitive Kartenspiele“**. Im Mittelpunkt steht das Kartenspiel **President**, das als Spiel mit unvollständiger Information in C++ für [OpenSpiel](https://github.com/google-deepmind/open_spiel) umgesetzt und über eine Python/PyTorch-Pipeline mit Multi-Agent Reinforcement Learning untersucht wird.

Das Projekt verbindet eine eigene Spielumgebung mit **PPO**, **DQN**, heuristischen Gegnern, Self-Play, Experiment Tracking und einer mehrstufigen Evaluation. Es bildet damit den vollständigen Ablauf von der Spielmodellierung und Zustandsrepräsentation bis zum Training und zur Analyse gelernter Policies ab.

> **Forschungsfrage:** Ist Reinforcement Learning geeignet, um in kompetitiven Kartenspielen wie President robuste Strategien zu entwickeln?

Der Schwerpunkt liegt auf einem methodischen Vergleich von Trainingskonzepten, Algorithmen, Trainingsgegnern und Reward Design — nicht auf maximaler Spielstärke durch umfassendes Hyperparameter-Tuning.

## Eigenleistung und Abgrenzung

Dieses Repository ist ein Research Fork von OpenSpiel. Das ursprüngliche OpenSpiel-Projekt stellt das allgemeine Game Framework, seine APIs, das Build System und zahlreiche Upstream-Algorithmen bereit. Die im Rahmen der Masterarbeit entstandenen Erweiterungen umfassen:

- die C++-Implementierung von President als registriertes OpenSpiel-Spiel;
- konfigurierbare Spielerzahlen, Deckgrößen, Legal Actions, Stichlogik und Platzierungsbewertung;
- Observation- und Information-State-Repräsentationen für die Nutzung über `pyspiel`;
- unabhängig von den OpenSpiel-Referenzalgorithmen entwickelte PyTorch-Implementierungen von PPO und DQN mit Legal-Action-Masking;
- heuristische und tabellarische Gegner sowie Population Training und Self-Play;
- Checkpointing, Run Metadata, Reward Shaping, Benchmarking und Plotting;
- ein reproduzierbares Analyse-Framework mit Macro- und Micro-Evaluation sowie der eigens entwickelten First-Action-Distribution.

Die eigene Spielimplementierung befindet sich unter [`open_spiel/games/president/`](open_spiel/games/president/). Die Trainings- und Evaluationspipeline der Masterarbeit liegt unter [`President/`](President/).

## Technischer Überblick

```text
President-Spiel (C++ / OpenSpiel)
              |
              v
      pyspiel Python binding
              |
              v
  Feature Encoding + Action Masking
              |
       +------+------+
       |             |
      PPO          DQN
       |             |
       +------+------+
              |
              v
 Checkpoints, Benchmarks, CSV-Auswertungen und Plots
```

### Spielumgebung

Die President-Umgebung modelliert ein sequenzielles Multiplayer-Kartenspiel mit unvollständiger Information. Sie unterstützt zwei bis acht Spieler, verschiedene Deckkonfigurationen, einzelne Karten und Kombinationen, Passen, Stichwechsel und eine platzierungsabhängige Bewertung. Die State Tensors enthalten die eigene Hand, die Handgrößen der Gegner, Informationen zum aktuellen Stich und die Historie bereits gespielter Karten.

### Agents und Trainingskonstellationen

| Familie | Algorithmus | Zentrale Komponenten |
|---|---|---|
| `a1` | PPO | Actor-Critic Networks, GAE, Clipped Updates, Entropy Regularization, Action Masking |
| `a2` | DQN | Replay Buffer, Target Network, Double-DQN-Option, Legal-Action Targets, konfigurierbare Exploration |

Die acht zentralen Trainingsskripte kombinieren beide Algorithmen mit vier Multi-Agent-Konstellationen:

| Konstellation | Trainingsaufbau | PPO | DQN |
|---|---|---|---|
| K1 | Single-Agent RL gegen feste Strategien oder Gegnerpopulationen | [`k1a1.py`](President/training/k1a1.py) | [`k1a2.py`](President/training/k1a2.py) |
| K2 | Independent Learning mit vier getrennten Agents | [`k2a1.py`](President/training/k2a1.py) | [`k2a2.py`](President/training/k2a2.py) |
| K3 | Snapshot Self-Play mit eingefrorenen Sparringspartnern | [`k3a1.py`](President/training/k3a1.py) | [`k3a2.py`](President/training/k3a2.py) |
| K4 | Shared-Policy Self-Play mit getrennter Datensammlung und Update-Phase | [`k4a1.py`](President/training/k4a1.py) | [`k4a2.py`](President/training/k4a2.py) |

Die Trainingskonfigurationen enthalten feste Random Seeds, konfigurierbares Reward Shaping, Opponent Pools, periodische Benchmarks, Model Checkpoints und Run Metadata.

### Evaluation

Die Evaluationspipeline ist vom Training getrennt und vergleicht gelernte Policies mit heuristischen, tabellarischen und weiteren gelernten Agents. Neben Winrate, Platzierungsverteilung und Average Return erfasst sie Verhaltensmerkmale wie Action-, Kombinationen-, Pass/Play-, Eröffnungs- und Rangverteilungen. Seat Balancing und Start Rotation reduzieren Positionsvorteile; die Hauptevaluation wertet 30.000 Episoden je Vergleichszelle deterministisch aus. Die Ausgaben werden als CSV-Zusammenfassungen, Logs und Plots gespeichert.

- [`eval_macro.py`](President/evaluation/eval_macro.py): aggregierte Auswertung über viele Spiele
- [`eval_micro.py`](President/evaluation/eval_micro.py): detaillierte Untersuchung einzelner Partien
- [`play_human.py`](President/evaluation/play_human.py): interaktives Spiel gegen konfigurierte Agents

## Zentrale Ergebnisse

Die Abgabeversion der Masterarbeit kommt innerhalb des untersuchten Settings zu folgenden Ergebnissen:

- Reinforcement Learning kann robuste Strategien für President mit unvollständiger Information erlernen.
- DQN ist im Gesamtvergleich überwiegend robuster und leistungsstärker als PPO; dieser Vorsprung hängt jedoch vom konkreten Setup ab.
- Die Multi-Agent- und Self-Play-Konzepte K2 bis K4 generalisieren besser als K1 gegen eine feste Heuristik. K1 erzielt hohe On-Target-Leistung, verliert Off-Target aber deutlich.
- Im 16-Karten-2v2-Setting erreichen die DQN-Varianten K2 bis K4 gegen den starken Referenzgegner V-Table Winrates knapp über 51 %, während die entsprechenden PPO-Varianten zwischen 43,8 % und 48,6 % liegen.
- Final Rewards genügen als primäres Lernsignal. Moderate Step Rewards können das Lernen unterstützen, starke oder isolierte Step Rewards verzerren dagegen die gelernten Strategien.

Die Ergebnisse sind als methodische Evidenz zu verstehen: Die Arbeit verwendet nur wenige unabhängige Seeds, minimales Hyperparameter-Tuning und eine deterministische Evaluation. Für das 64-Karten-Setting fehlen zudem eine mit V-Table vergleichbare starke Referenz und ein Human Benchmark.

## Projektstruktur

```text
.
├── open_spiel/games/president/   # Eigene President-Implementierung in C++
├── President/
│   ├── agents/                   # PPO, DQN und tabellarische Agents
│   ├── training/                 # Trainingskonfigurationen K1 bis K4
│   ├── evaluation/               # Macro-, Micro- und interaktive Evaluation
│   ├── utils/                    # Features, Rewards, Benchmarks, Plots und I/O
│   └── models/                   # Checkpoints und Run-Artefakte der Masterarbeit
├── requirements/                 # Python-Abhängigkeiten und Constraints
├── Playground/                   # Historische Prototypen und explorative Runs
└── open_spiel/                   # OpenSpiel-Framework und Upstream-Quellen
```

## Quickstart

Der folgende Ablauf entspricht dem erfolgreich getesteten lokalen Linux-Build mit **Python 3.12**, **Clang**, **CMake** und **PyTorch 2.7.1 (CPU)**. Benötigt werden außerdem Git, die Python-3.12-Entwicklungsdateien und das `venv`-Modul. OpenSpiel setzt mindestens CMake 3.17 und Clang 7 voraus.

### 1. Externe OpenSpiel-Build-Abhängigkeiten

Die folgenden, von OpenSpiel erwarteten C++-Quellen sind bewusst nicht im Git-Repository enthalten. Die Revisionen entsprechen dem getesteten lokalen Build:

```bash
git clone https://github.com/pybind/pybind11.git pybind11
git -C pybind11 checkout 1c10d5e9b1e0cecf7401af735bb3bc515043760d

git clone --branch 20230125.0 --depth 1 \
  https://github.com/abseil/abseil-cpp.git open_spiel/abseil-cpp

git clone https://github.com/pybind/pybind11_abseil.git open_spiel/pybind11_abseil
git -C open_spiel/pybind11_abseil checkout 73992b54e7d40dfada5e5cd998cad60917f0b3d1

git clone --branch develop \
  https://github.com/jblespiau/dds.git open_spiel/games/bridge/double_dummy_solver
git -C open_spiel/games/bridge/double_dummy_solver checkout 091ea94358a4016d4fb6069dea5c452cdc98d0bd
```

### 2. Python-Umgebung

Ein Virtual Environment anlegen und die mit Constraints dokumentierten Python-Abhängigkeiten installieren:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  --constraint requirements/constraints.txt torch==2.7.1
python -m pip install \
  --constraint requirements/constraints.txt \
  --requirement requirements/base.txt
```

Falls der lokale `pip`-Cache Probleme verursacht, kann den Installationsbefehlen optional `--no-cache-dir` hinzugefügt werden.

### 3. Lokaler CMake-Build

Den OpenSpiel Fork mit Clang konfigurieren und ausschließlich das lokale `pyspiel`-Binding bauen:

```bash
cmake -S open_spiel -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v clang++)" \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build --target pyspiel --parallel 2
```

### 4. Smoke Test

Die Repository-Module und das kompilierte Binding über `PYTHONPATH` verfügbar machen und President mit den im C++-Code registrierten Standardparametern laden:

```bash
export PYTHONPATH="$PWD:$PWD/build/python${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import pyspiel

game = pyspiel.load_game("president")
state = game.new_initial_state()

print(game)
print(state)
print("Legal actions:", state.legal_actions())
PY
```

Der Default entspricht vier Spielern, einem 32-Karten-Deck, gemischten Karten und aktivierten Kartenkombinationen. Evaluation und Plotting benötigen zusätzlich:

```bash
python -m pip install -r requirements/evaluation.txt
```

Die zentralen Trainingsskripte sind für umfangreiche Experimente der Masterarbeit konfiguriert. Vor der Ausführung sollten die jeweiligen `CONFIG` Dictionaries geprüft werden, da Episodenzahlen und Ausgabeverzeichnisse je nach Experimentfamilie variieren.

## Attribution und Lizenz

Dieses Projekt basiert auf **OpenSpiel**, entwickelt von Google DeepMind und externen Contributors:

- [OpenSpiel Source Repository](https://github.com/google-deepmind/open_spiel)
- [OpenSpiel Documentation](https://openspiel.readthedocs.io/)
- Marc Lanctot et al., [*OpenSpiel: A Framework for Reinforcement Learning in Games*](https://arxiv.org/abs/1908.09453), 2019

Das Upstream-Framework und dieses Repository werden unter der [Apache License 2.0](LICENSE) bereitgestellt. Die President-Umgebung und die Python-Pipeline der Masterarbeit sind projektspezifische Erweiterungen. Es besteht keine organisatorische Zugehörigkeit zu Google DeepMind; eine Empfehlung durch Google DeepMind ist damit nicht verbunden.
