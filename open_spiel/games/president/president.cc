#include "open_spiel/games/president/president.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"

namespace open_spiel {
namespace president {
namespace {

// Game registration
const GameType kGameType{
    /*short_name=*/"president",
    /*long_name=*/"President",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/6,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/false,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"shuffle_cards", GameParameter(true)},
        {"single_card_mode", GameParameter(true)},
        {"start_player_mode", GameParameter(std::string("fixed"))},
    }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<PresidentGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Utility string conversion helpers
std::string RankToString(int rank) {
  static const std::vector<std::string> kRanks = {"7", "8", "9", "10", "J", "Q", "K", "A"};
  return kRanks[rank];
}

std::string ComboToString(ComboType type) {
  static const std::vector<std::string> kCombos = {"Single", "Pair", "Triple", "Quad"};
  return kCombos[static_cast<int>(type)];
}

std::string HandToString(const Hand& hand) {
  std::string result;
  for (int r = 0; r < kNumRanks; ++r) {
    for (int i = 0; i < hand[r]; ++i) {
      absl::StrAppend(&result, RankToString(r), " ");
    }
  }
  return result;
}

std::vector<int> LegalActionsFromHand(const Hand& hand, int top_rank,
                                      ComboType current_type, bool new_trick,
                                      bool single_card_mode) {
  std::vector<int> actions = {0};  // Pass
  int max_combo_size = single_card_mode ? 1 : 4;

  for (int rank = 0; rank < kNumRanks; ++rank) {
    int count = hand[rank];
    for (int size = 1; size <= std::min(count, max_combo_size); ++size) {
      ComboType type = static_cast<ComboType>(static_cast<int>(ComboType::Single) + size - 1);
      if (new_trick || (type == current_type && rank > top_rank)) {
        actions.push_back(EncodeAction({type, rank}));
      }
    }
  }
  return actions;
}

void ApplyPresidentAction(const PresidentAction& action, Hand& hand) {
  if (action.type == ComboType::Pass) return;
  int count = static_cast<int>(action.type) - static_cast<int>(ComboType::Single) + 1;
  hand[action.rank] -= count;
}

}  // namespace

// PresidentGame implementation
PresidentGame::PresidentGame(const GameParameters& params)
    : Game(kGameType, params),
      shuffle_cards_(ParameterValue<bool>("shuffle_cards")),
      single_card_mode_(ParameterValue<bool>("single_card_mode")),
      rotate_index_(0),
      last_loser_(std::nullopt) {
  std::string mode = ParameterValue<std::string>("start_player_mode");
  if (mode == "fixed") start_mode_ = StartPlayerMode::Fixed;
  else if (mode == "random") start_mode_ = StartPlayerMode::Random;
  else if (mode == "rotate") start_mode_ = StartPlayerMode::Rotate;
  else if (mode == "loser") start_mode_ = StartPlayerMode::Loser;
  else SpielFatalError("Unknown start mode: " + mode);
}

std::unique_ptr<State> PresidentGame::NewInitialState() const {
  return std::make_unique<PresidentGameState>(shared_from_this(), shuffle_cards_);
}

// PresidentGameState implementation
PresidentGameState::PresidentGameState(std::shared_ptr<const Game> game, bool shuffle)
    : State(game),
      num_players_(4),
      current_player_(0),
      last_player_to_play_(-1),
      top_rank_(-1),
      current_combo_type_(ComboType::Pass),
      new_trick_(true),
      hands_(num_players_, Hand(kNumRanks, 0)),
      passed_(num_players_, false),
      finish_order_(),
      single_card_mode_(std::static_pointer_cast<const PresidentGame>(game)->single_card_mode_),
      start_mode_(std::static_pointer_cast<const PresidentGame>(game)->start_mode_),
      rotate_start_index_(std::static_pointer_cast<const PresidentGame>(game)->rotate_index_) {

  std::vector<int> deck;
  for (int r = 0; r < kNumRanks; ++r)
    for (int i = 0; i < 4; ++i)
      deck.push_back(r);

  if (shuffle) {
    std::shuffle(deck.begin(), deck.end(), std::mt19937(std::random_device{}()));
  }

  for (int i = 0; i < deck.size(); ++i) {
    hands_[i % num_players_][deck[i]]++;
  }

  const auto* pg = static_cast<const PresidentGame*>(game_.get());
  switch (start_mode_) {
    case StartPlayerMode::Fixed:
      current_player_ = 0;
      break;
    case StartPlayerMode::Random: {
      std::mt19937 rng(std::random_device{}());
      std::uniform_int_distribution<int> dist(0, num_players_ - 1);
      current_player_ = dist(rng);
      break;
    }
    case StartPlayerMode::Rotate:
      current_player_ = rotate_start_index_;
      break;
    case StartPlayerMode::Loser:
      current_player_ = pg->last_loser_.value_or(0);
      break;
  }

}

Player PresidentGameState::CurrentPlayer() const {
  return current_player_;
}

std::vector<Action> PresidentGameState::LegalActions() const {
  auto raw = LegalActionsFromHand(hands_[current_player_], top_rank_, current_combo_type_, new_trick_, single_card_mode_);
  return std::vector<Action>(raw.begin(), raw.end());
}

std::string PresidentGameState::ActionToString(Player, Action action_id) const {
  PresidentAction action = DecodeAction(action_id);
  if (action.type == ComboType::Pass) return "Pass";
  return absl::StrFormat("Play %s of %s", ComboToString(action.type), RankToString(action.rank));
}

std::string PresidentGameState::ToString() const {
  std::string out = absl::StrFormat("Player %d to play\n", current_player_);
  for (int i = 0; i < num_players_; ++i) {
    absl::StrAppend(&out, "P", i, ": ", HandToString(hands_[i]));
    if (passed_[i]) absl::StrAppend(&out, "(pass)");
    if (std::find(finish_order_.begin(), finish_order_.end(), i) != finish_order_.end())
      absl::StrAppend(&out, "(done)");
    absl::StrAppend(&out, "\n");
  }
  if (top_rank_ >= 0) {
    absl::StrAppend(&out, "Current trick: ", RankToString(top_rank_));
    if (new_trick_) absl::StrAppend(&out, " (new)");
    absl::StrAppend(&out, " (played by P", last_player_to_play_, ")\n");
  }
  return out;
}

bool PresidentGameState::IsTerminal() const {
  return finish_order_.size() >= num_players_ - 1;
}

std::vector<double> PresidentGameState::Returns() const {
  std::vector<double> scores(num_players_, 0.0);
  const std::vector<double> points = {3, 2, 1};
  for (int i = 0; i < finish_order_.size() && i < points.size(); ++i) {
    scores[finish_order_[i]] = points[i];
  }

  auto* game = const_cast<PresidentGame*>(static_cast<const PresidentGame*>(game_.get()));
  if (start_mode_ == StartPlayerMode::Loser && finish_order_.size() == num_players_ - 1) {
    for (int p = 0; p < num_players_; ++p) {
      if (!IsOut(p)) game->last_loser_ = p;
    }
  }
  if (start_mode_ == StartPlayerMode::Rotate) {
    game->rotate_index_ = (rotate_start_index_ + 1) % num_players_;
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
    if (IsOut(current_player_) &&
        std::find(finish_order_.begin(), finish_order_.end(), current_player_) == finish_order_.end()) {
      finish_order_.push_back(current_player_);
    }
    last_player_to_play_ = current_player_;
    top_rank_ = action.rank;
    current_combo_type_ = action.type;
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
    current_combo_type_ = ComboType::Pass;
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

void PresidentGameState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kNumRanks);
  for (int i = 0; i < kNumRanks; ++i) {
    values[i] = static_cast<float>(hands_[player][i]);
  }
}

void PresidentGameState::InformationStateTensor(Player player, absl::Span<float> values) const {
  ObservationTensor(player, values);
}

}  // namespace president
}  // namespace open_spiel
