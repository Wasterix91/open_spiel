#ifndef OPEN_SPIEL_GAMES_PRESIDENT_H_
#define OPEN_SPIEL_GAMES_PRESIDENT_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace president {


enum class ComboType { Pass, Play };

// Ein Spielzug: Pass oder Spielzug mit combo_size >= 1
struct PresidentAction {
  ComboType type;  // Pass oder Spielzug
  int combo_size;  // Nur für Spielzug: wie viele gleiche Karten (1-8)
  int rank;        // Kartenrang

  bool operator==(const PresidentAction& other) const {
    return type == other.type && combo_size == other.combo_size && rank == other.rank;
  }
};

// Kodierung: Action -> int
inline int EncodeAction(const PresidentAction& action, int num_ranks) {
  if (action.type == ComboType::Pass) return 0;
  return 1 + (action.combo_size - 1) * num_ranks + action.rank;
}

// Dekodierung: int -> Action
inline PresidentAction DecodeAction(int id, int num_ranks) {
  if (id == 0) return {ComboType::Pass, 0, -1};
  int adjusted = id - 1;
  int combo_size = 1 + adjusted / num_ranks;
  int rank = adjusted % num_ranks;
  return {ComboType::Play, combo_size, rank};  
}

// Eine Hand = Vektor mit Kartenanzahlen pro Rang
using Hand = std::vector<int>;

// Startspieler-Strategie
enum class StartPlayerMode { Fixed, Random, Rotate, Loser };

// Forward declaration
class PresidentGame;

// -----------------------------
// State-Klasse
// -----------------------------
class PresidentGameState : public State {
 public:
  PresidentGameState(std::shared_ptr<const Game> game, bool shuffle);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  void ApplyAction(Action action_id) override;

  void ObservationTensor(Player player, absl::Span<float> values) const override;
  void InformationStateTensor(Player player, absl::Span<float> values) const override;

 private:
  void AdvanceToNextPlayer();
  bool IsOut(int player) const;

  int num_players_;
  int current_player_;
  int last_player_to_play_;
  int top_rank_;
  int current_combo_size_;  // NEU: statt enum
  bool new_trick_;
  std::vector<Hand> hands_;
  std::vector<bool> passed_;
  std::vector<int> finish_order_;
  bool single_card_mode_;

  StartPlayerMode start_mode_;
  int rotate_start_index_;

  int kNumRanks;
  std::vector<std::string> ranks_;
};

// -----------------------------
// Game-Klasse
// -----------------------------
class PresidentGame : public Game {
 public:
  explicit PresidentGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override { return 1 + kMaxComboSize * kNumRanks; }
  int MaxChanceOutcomes() const override { return 0; }
  int NumPlayers() const override { return 4; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return 3; }
  int MaxGameLength() const override { return 100; }
  std::vector<int> ObservationTensorShape() const override { return {kNumRanks}; }
  std::vector<int> InformationStateTensorShape() const override { return {kNumRanks}; }

  StartPlayerMode start_mode_;
  int rotate_index_;
  std::optional<int> last_loser_;

  int kNumRanks;
  int kMaxComboSize;  // NEU: dynamisch bestimmt aus Deckgröße
  int num_suits_;
  std::vector<std::string> ranks_;

 private:
  bool shuffle_cards_;
  bool single_card_mode_;

  friend class PresidentGameState;
};

}  // namespace president
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PRESIDENT_H_
