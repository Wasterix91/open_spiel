#ifndef OPEN_SPIEL_GAMES_PRESIDENT_H_
#define OPEN_SPIEL_GAMES_PRESIDENT_H_

#include <memory>
#include <string>
#include <vector>
#include <optional>

#include "open_spiel/spiel.h"

namespace open_spiel {
namespace president {

using Hand = std::vector<int>;

enum class ComboType { Pass, Play };

struct PresidentAction {
  ComboType type;
  int combo_size;
  int rank;
};

// Encoding / Decoding
inline int EncodeAction(const PresidentAction& action, int num_ranks) {
  if (action.type == ComboType::Pass) return 0;
  return 1 + (action.combo_size - 1) * num_ranks + action.rank;
}

inline PresidentAction DecodeAction(int action_id, int num_ranks) {
  if (action_id == 0) return {ComboType::Pass, 0, 0};
  int tmp = action_id - 1;
  int size = tmp / num_ranks + 1;
  int rank = tmp % num_ranks;
  return {ComboType::Play, size, rank};
}

// === Game ===
class PresidentGame : public Game {
 public:
  explicit PresidentGame(const GameParameters& params);

  // === Required overrides ===
  int NumPlayers() const override;
  int NumDistinctActions() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  int MaxGameLength() const override;
  std::unique_ptr<State> NewInitialState() const override;

  // === Tensor shapes ===
  std::vector<int> ObservationTensorShape() const override;
  std::vector<int> InformationStateTensorShape() const override;

  // === Game Parameters ===
  int num_players_;
  bool shuffle_cards_;
  bool single_card_mode_;
  //enum class StartPlayerMode { Fixed, Random, Rotate, Loser } start_mode_;
  //int rotate_index_;
  //std::optional<int> last_loser_;

  std::vector<std::string> ranks_;
  int kNumRanks;
  int kMaxComboSize;
  int num_suits_;
};

// === State ===
class PresidentGameState : public State {
 public:
  explicit PresidentGameState(std::shared_ptr<const Game> game, bool shuffle, int start_player);

  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::unique_ptr<State> Clone() const override;
  void ApplyAction(Action action_id) override;
  std::vector<int> GetFinishOrder();

  // === Observation und InformationState ===
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  std::string ObservationString(Player player) const override;

  void InformationStateTensor(Player player, absl::Span<float> values) const override;
  std::string InformationStateString(Player player) const override;

 private:
  int current_player_;
  int last_player_to_play_;
  int top_rank_;
  int current_combo_size_;
  int rotate_start_index_;

  std::vector<Hand> hands_;
  //std::vector<bool> passed_;
  std::vector<int> finish_order_;

  bool IsOut(int player) const;
};

}  // namespace president
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PRESIDENT_H_
