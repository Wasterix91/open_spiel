#include "open_spiel/games/president/president.h"

#include <numeric>
#include <algorithm>
#include <random>
#include <memory>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace president {

// ---------- Hilfsfunktionen -------------

std::vector<int> LegalActionsFromHand(const Hand& hand, int top_rank, bool new_trick) {
  std::vector<int> actions = {0};  // Pass immer erlaubt
  for (int i = 0; i < kNumRanks; ++i) {
    if (hand[i] > 0 && (new_trick || i > top_rank)) {
      actions.push_back(1 + i);
    }
  }
  return actions;
}

void ApplyPresidentAction(const PresidentAction& action, Hand& hand) {
  if (action.type != ComboType::Pass) {
    hand[action.rank]--;
  }
}

std::string HandToString(const Hand& hand) {
  static const std::vector<std::string> ranks = {"7", "8", "9", "10", "U", "O", "K", "A"};
  std::string result;
  for (int i = 0; i < kNumRanks; ++i) {
    for (int j = 0; j < hand[i]; ++j) {
      result += ranks[i] + " ";
    }
  }
  return result;
}

// ---------- Spielzustand -------------

PresidentGameState::PresidentGameState(std::shared_ptr<const Game> game, bool shuffle)
    : State(game),
      num_players_(4),
      current_player_(0),
      last_player_to_play_(-1),
      top_rank_(-1),
      new_trick_(true),
      hands_(num_players_, Hand(kNumRanks, 0)),
      passed_(num_players_, false),
      finish_order_() {

  std::vector<int> deck;
  for (int rank = 0; rank < kNumRanks; ++rank) {
    for (int count = 0; count < 4; ++count) {
      deck.push_back(rank);
    }
  }

  if (shuffle) {
    std::shuffle(deck.begin(), deck.end(), std::mt19937(std::random_device{}()));
  }

  for (int i = 0; i < deck.size(); ++i) {
    int player = i % num_players_;
    hands_[player][deck[i]]++;
  }
}

Player PresidentGameState::CurrentPlayer() const { return current_player_; }

std::vector<Action> PresidentGameState::LegalActions() const {
  std::vector<int> raw = LegalActionsFromHand(hands_[current_player_], top_rank_, new_trick_);
  return std::vector<Action>(raw.begin(), raw.end());
}

std::string PresidentGameState::ActionToString(Player, Action action_id) const {
  PresidentAction action = DecodeAction(action_id);
  if (action.type == ComboType::Pass) return "Pass";
  static const std::vector<std::string> ranks = {"7", "8", "9", "10", "U", "O", "K", "A"};
  return "Play " + ranks[action.rank];
}

std::string PresidentGameState::ToString() const {
  std::string s = "Spieler " + std::to_string(current_player_) + " am Zug\n";
  for (int i = 0; i < num_players_; ++i) {
    s += "P" + std::to_string(i) + ": " + HandToString(hands_[i]);
    if (passed_[i]) s += "(pass)";
    if (std::find(finish_order_.begin(), finish_order_.end(), i) != finish_order_.end()) {
      s += "(fertig)";
    }
    s += "\n";
  }

  if (top_rank_ >= 0) {
    static const std::vector<std::string> ranks = {"7", "8", "9", "10", "U", "O", "K", "A"};
    s += "Current trick: " + ranks[top_rank_];
    if (new_trick_) s += " (new)";
    s += " (played by P" + std::to_string(last_player_to_play_) + ")";
    s += "\n";
  }

  return s;
}

bool PresidentGameState::IsTerminal() const {
  return finish_order_.size() >= num_players_ - 1;
}

std::vector<double> PresidentGameState::Returns() const {
  std::vector<double> scores(num_players_, 0);
  const std::vector<double> points = {3, 2, 1};
  for (int i = 0; i < finish_order_.size() && i < points.size(); ++i) {
    scores[finish_order_[i]] = points[i];
  }
  return scores;
}

std::unique_ptr<State> PresidentGameState::Clone() const {
  return std::make_unique<PresidentGameState>(*this);
}

void PresidentGameState::ApplyAction(Action action_id) {
  PresidentAction action = DecodeAction(action_id);

  if (action.type == ComboType::Pass) {
    passed_[current_player_] = true;
  } else {
    ApplyPresidentAction(action, hands_[current_player_]);

    if (std::accumulate(hands_[current_player_].begin(), hands_[current_player_].end(), 0) == 0 &&
        std::find(finish_order_.begin(), finish_order_.end(), current_player_) == finish_order_.end()) {
      finish_order_.push_back(current_player_);
    }

    last_player_to_play_ = current_player_;
    top_rank_ = action.rank;
    new_trick_ = false;
    std::fill(passed_.begin(), passed_.end(), false);
  }

  AdvanceToNextPlayer();
}

void PresidentGameState::AdvanceToNextPlayer() {
  int active = 0;
  for (int i = 0; i < num_players_; ++i) {
    if (!passed_[i] && !IsOut(i)) ++active;
  }

  if (active == 1 && last_player_to_play_ != -1) {
    current_player_ = last_player_to_play_;
    top_rank_ = -1;
    new_trick_ = true;
    passed_ = std::vector<bool>(num_players_, false);
  } else {
    do {
      current_player_ = (current_player_ + 1) % num_players_;
    } while (IsOut(current_player_));
  }
}

bool PresidentGameState::IsOut(int player) const {
  return std::accumulate(hands_[player].begin(), hands_[player].end(), 0) == 0;
}

// ---------- Game -------------

const GameType kGameType{
    /* short_name = */ "president",
    /* long_name = */ "President",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /* max_num_players = */ 6,
    /* min_num_players = */ 2,
    /* provides_information_state_string = */ false,
    /* provides_information_state_tensor = */ false,
    /* provides_observation_string = */ true,
    /* provides_observation_tensor = */ false,
    /* parameter_specification = */
    {{"shuffle_cards", GameParameter(true)}}
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<PresidentGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

PresidentGame::PresidentGame(const GameParameters& params)
    : Game(kGameType, params),
      shuffle_cards_(ParameterValue<bool>("shuffle_cards")) {}

std::unique_ptr<State> PresidentGame::NewInitialState() const {
  return std::make_unique<PresidentGameState>(shared_from_this(), shuffle_cards_);
}

}  // namespace president
}  // namespace open_spiel
