#ifndef OPEN_SPIEL_GAMES_PRESIDENT_H_
#define OPEN_SPIEL_GAMES_PRESIDENT_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace president {

// Number of ranks in President (7 to Ace).
inline constexpr int kNumRanks = 8;

// Maximum number of unique actions (1 Pass + 4*8 combinations).
inline constexpr int kNumActionTypes = 4;
inline constexpr int kNumActions = 1 + kNumRanks * kNumActionTypes;

// Types of combinations that can be played.
enum class ComboType { Pass, Single, Pair, Triple, Quad };

// Struct representing a game action (combination type and rank).
struct PresidentAction {
  ComboType type;
  int rank;

  bool operator==(const PresidentAction& other) const {
    return type == other.type && rank == other.rank;
  }
};

// Encodes a PresidentAction into a unique integer.
inline int EncodeAction(const PresidentAction& action) {
  if (action.type == ComboType::Pass) return 0;
  return 1 + (static_cast<int>(action.type) - static_cast<int>(ComboType::Single)) * kNumRanks + action.rank;
}

// Decodes an integer into a PresidentAction.
inline PresidentAction DecodeAction(int id) {
  if (id == 0) return {ComboType::Pass, -1};
  int adjusted = id - 1;
  ComboType type = static_cast<ComboType>(static_cast<int>(ComboType::Single) + adjusted / kNumRanks);
  int rank = adjusted % kNumRanks;
  return {type, rank};
}

// A hand is a vector representing how many cards of each rank a player has.
using Hand = std::vector<int>;

// Start player logic for new rounds.
enum class StartPlayerMode { Fixed, Random, Rotate, Loser };

// Forward declaration.
class PresidentGame;

// Game state class.
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
  ComboType current_combo_type_;
  bool new_trick_;
  std::vector<Hand> hands_;
  std::vector<bool> passed_;
  std::vector<int> finish_order_;
  bool single_card_mode_;

  StartPlayerMode start_mode_;
  int rotate_start_index_;
};

// Game class.
class PresidentGame : public Game {
 public:
  explicit PresidentGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override { return kNumActions; }
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

 private:
  bool shuffle_cards_;
  bool single_card_mode_;

  friend class PresidentGameState;
};

}  // namespace president
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PRESIDENT_H_
