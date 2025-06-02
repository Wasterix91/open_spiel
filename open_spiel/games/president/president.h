#ifndef OPEN_SPIEL_GAMES_PRESIDENT_H_
#define OPEN_SPIEL_GAMES_PRESIDENT_H_

#include <string>
#include <vector>
#include <memory>
#include <optional>  // Für std::optional
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace president {

enum class ComboType {
  Pass,
  Single,
  Pair,
  Triple,
  Quad,
};

// Kombination von 1 bis 4 Karten gleichen Ranges
struct PresidentAction {
  ComboType type;
  int rank;

  bool operator==(const PresidentAction& other) const {
    return type == other.type && rank == other.rank;
  }
};

inline constexpr int kNumRanks = 8; // 7,8,9,10,U,O,K,A
inline constexpr int kNumActionTypes = 4; // Single, Pair, Triple, Quad
inline constexpr int kNumActions = 1 + kNumRanks * kNumActionTypes; // Anzahl möglicher Aktionen

// Kodierung der Aktionen. Pass -> 0; Single -> 1-8; Pair -> 9-16; Triple -> 17-24; Quad -> 25-32
inline int EncodeAction(const PresidentAction& action) {
  if (action.type == ComboType::Pass) return 0;
  return 1 + (static_cast<int>(action.type) - static_cast<int>(ComboType::Single)) * kNumRanks + action.rank;
}

inline PresidentAction DecodeAction(int id) {
  if (id == 0) return PresidentAction{ComboType::Pass, -1};
  int adjusted = id - 1;
  ComboType type = static_cast<ComboType>(static_cast<int>(ComboType::Single) + adjusted / kNumRanks);
  int rank = adjusted % kNumRanks;
  return PresidentAction{type, rank};
}

// Vektor aus Anzahl * Rang, 
// Beispiel: Hand player_hand = {1,0,2,0,0,1,0,0} 
// bedeutet: {1*7, 0*8, 2*9, 0*10, 0*U, 1*O, 0*K, 0*A}
using Hand = std::vector<int>;

// berechnet legale Aktionen abhängig von 
// top_rank (Höchste gespielte Karte)
// current_type (Kombinationsart)
// new_trick (Hat neuer Trick begonnen?)
// single_card_mode (Nur Einzelkarten erlaubt?)
std::vector<int> LegalActionsFromHand(const Hand& hand, int top_rank, ComboType current_type, bool new_trick, bool single_card_mode);

// Entfernt gespielte Karte aus Hand
void ApplyPresidentAction(const PresidentAction& action, Hand& hand);

// Lesbarer String der Hand "2x Rank 0, 1x Rank 5"
std::string HandToString(const Hand& hand);

// Mögliche Modi, wie bestimmt wird, wer die nächste Runde beginnt
enum class StartPlayerMode { Fixed, Random, Rotate, Loser };

class PresidentGame;

// Aktueller Zustand des Spiels
class PresidentGameState : public open_spiel::State {
 public:
  PresidentGameState(std::shared_ptr<const Game> game, bool shuffle);

  // Öffentliche Methoden: OpenSpiel-Funktionen
  Player CurrentPlayer() const override; // Wer ist am Zug?
  std::vector<Action> LegalActions() const override; // Welche Züge sind erlaubt?
  std::string ActionToString(Player player, Action action_id) const override; 
  std::string ToString() const override; // Ganzer Spielzustand als String
  bool IsTerminal() const override; // Spielende?
  std::vector<double> Returns() const override; // Scores für alle Spieler
  std::unique_ptr<State> Clone() const override; // Kopie des Spielzustands
  void ApplyAction(Action action_id) override; // Ändert den Zustand

  void ObservationTensor(Player player, absl::Span<float> values) const override; // RL: Beobachtung des Spielzustands
  void InformationStateTensor(Player player, absl::Span<float> values) const override; // RL: private Infos pro Spieler

 private:
  void AdvanceToNextPlayer(); // Schaltet weiter zu nächstem Spieler
  bool IsOut(int player) const; // Überprüft, ob Spieler fertig

  // Interne Variablen - was wird gespeichert?
  int num_players_; // Anzahl der Spieler
  int current_player_; // Wer ist am Zug?
  int last_player_to_play_; // Wer hat als letztes gespielt?
  int top_rank_; // Rang der aktuell höchstgespielten Karte
  ComboType current_combo_type_; // Welche Kombinationsart liegt aktuell aus
  bool new_trick_; // Ist neue Spielrunde gestartet?
  std::vector<Hand> hands_; // Kartenhände der Spieler
  std::vector<bool> passed_; // Wer hat gepasst?
  std::vector<int> finish_order_; // Wer hat Spiel in welcher Reihenfolge beendet
  bool single_card_mode_; // Spielmodus

  // Startspieler-Logik
  StartPlayerMode start_mode_;           // Modus: fixed, random, rotate, loser
  int rotate_start_index_;               // aktueller Index für "rotate"
};

// Beschreibt das Spiel selbst
class PresidentGame : public Game {
 public:
  explicit PresidentGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override; // Anfangszustand
  int NumDistinctActions() const override { return kNumActions; } // 1 Pass + 4*8 Kombinationen
  int MaxChanceOutcomes() const override { return 0; } // Keine Zufallskomponenten
  std::vector<int> ObservationTensorShape() const override { return {kNumRanks}; } // Form für Beobachtung
  std::vector<int> InformationStateTensorShape() const override { return {kNumRanks}; } // Form für Informationszustand
  double MinUtility() const override { return 0; } // Minimale Punktezahl
  double MaxUtility() const override { return 3; } // Maximale Punktezahl
  int NumPlayers() const override { return 4; } // Spielerzahl ist fest
  int MaxGameLength() const override { return 100; } // Maximal 100 Züge

  // Startspieler-Logik
  StartPlayerMode start_mode_;          // Gewählter Modus
  int rotate_index_ = 0;                // Index für Rotationsstart
  std::optional<int> last_loser_;       // Letzter Verlierer für Loser-Modus

 private:
  bool shuffle_cards_; // Wird am Anfang gemischt?
  bool single_card_mode_; // Nur Einzelkarten?
  friend class PresidentGameState;
};

}  // namespace president
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PRESIDENT_H_
