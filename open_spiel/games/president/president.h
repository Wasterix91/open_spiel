#ifndef OPEN_SPIEL_GAMES_PRESIDENT_H_
#define OPEN_SPIEL_GAMES_PRESIDENT_H_

#include <string>
#include <vector>
#include <memory>
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace president {

enum class ComboType {
  Pass,
  Single,
};

struct PresidentAction {
  ComboType type;
  int rank;

  bool operator==(const PresidentAction& other) const {
    return type == other.type && rank == other.rank;
  }
};

inline int EncodeAction(const PresidentAction& action) {
  if (action.type == ComboType::Pass) return 0;
  return 1 + action.rank;
}

inline PresidentAction DecodeAction(int id) {
  if (id == 0) return PresidentAction{ComboType::Pass, -1};
  return PresidentAction{ComboType::Single, id - 1};
}

inline constexpr int kNumRanks = 8;
inline constexpr int kNumActions = 1 + kNumRanks;

using Hand = std::vector<int>;

std::vector<int> LegalActionsFromHand(const Hand& hand, int top_rank, bool new_trick);
void ApplyPresidentAction(const PresidentAction& action, Hand& hand);
std::string HandToString(const Hand& hand);

class PresidentGame;

class PresidentGameState : public open_spiel::State {
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

 private:
  void AdvanceToNextPlayer();
  bool IsOut(int player) const;

  int num_players_;
  int current_player_;
  int last_player_to_play_;
  int top_rank_;
  bool new_trick_;
  std::vector<Hand> hands_;
  std::vector<bool> passed_;
  std::vector<int> finish_order_;
};

class PresidentGame : public Game {
 public:
  explicit PresidentGame(const GameParameters& params);

  std::unique_ptr<State> NewInitialState() const override;
  int NumDistinctActions() const override { return kNumActions; }
  int MaxChanceOutcomes() const override { return 0; }
  std::vector<int> ObservationTensorShape() const override { return {}; }
  std::vector<int> InformationStateTensorShape() const override { return {}; }
  double MinUtility() const override { return 0; }
  double MaxUtility() const override { return 3; }
  int NumPlayers() const override { return 4; }
  int MaxGameLength() const override { return 100; }

 private:
  bool shuffle_cards_;
  friend class PresidentGameState;
};

}  // namespace president
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_PRESIDENT_H_
