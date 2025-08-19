#include "open_spiel/games/president/president.h"

#include <algorithm>
#include <iostream>
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

// === Game registration ===
const GameType kGameType{
    /*short_name=*/"president",
    /*long_name=*/"President",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kSampledStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/8,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"num_players", GameParameter(4)},  // ✅ Dynamisch
        {"shuffle_cards", GameParameter(true)},
        {"single_card_mode", GameParameter(false)},
        {"deck_size", GameParameter(std::string("32"))},
    }
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<PresidentGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string RankToString(int rank, const std::vector<std::string>& ranks) {
  return ranks[rank];
}

std::string ComboToString(int size) {
  if (size == 1) return "Single";
  if (size == 2) return "Pair";
  if (size == 3) return "Triple";
  if (size == 4) return "Quad";
  return absl::StrFormat("%d-of-a-kind", size);
}

std::string HandToString(const Hand& hand, const std::vector<std::string>& ranks) {
  std::string result;
  for (int r = 0; r < hand.size(); ++r) {
    for (int i = 0; i < hand[r]; ++i) {
      absl::StrAppend(&result, RankToString(r, ranks), " ");
    }
  }
  return result;
}

void ApplyPresidentAction(const PresidentAction& action, Hand& hand) {
  if (action.type == ComboType::Pass) return;
  hand[action.rank] -= action.combo_size;
}

}  // namespace

// === PresidentGame ===
PresidentGame::PresidentGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("num_players")),
      shuffle_cards_(ParameterValue<bool>("shuffle_cards")),
      single_card_mode_(ParameterValue<bool>("single_card_mode"))
      //rotate_index_(0), last_loser_(std::nullopt)
      {

  std::string deck_size_str = ParameterValue<std::string>("deck_size");
  if (deck_size_str == "32") {
    ranks_ = {"7", "8", "9", "10", "J", "Q", "K", "A"};
    num_suits_ = 4;
  } else if (deck_size_str == "52") {
    ranks_ = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
    num_suits_ = 4;
  } else if (deck_size_str == "64") {
    ranks_ = {"7", "8", "9", "10", "J", "Q", "K", "A"};
    num_suits_ = 8;
  } else if (deck_size_str == "12") {
    ranks_ = {"Q", "K", "A"};
    num_suits_ = 4;
  } else if (deck_size_str == "16") {
    ranks_ = {"J", "Q", "K", "A"};
    num_suits_ = 4;
  } else if (deck_size_str == "20") {
    ranks_ = {"10", "J", "Q", "K", "A"};
    num_suits_ = 4;
  } else if (deck_size_str == "24") {
    ranks_ = {"9", "10", "J", "Q", "K", "A"};
    num_suits_ = 4;
  }
  else {
    SpielFatalError("Unknown deck_size: " + deck_size_str);
  }

  kNumRanks = ranks_.size();
  kMaxComboSize = num_suits_;

  /*
  std::string mode = ParameterValue<std::string>("start_player_mode");
  if (mode == "fixed") start_mode_ = StartPlayerMode::Fixed;
  else if (mode == "random") start_mode_ = StartPlayerMode::Random;
  else if (mode == "rotate") start_mode_ = StartPlayerMode::Rotate;
  else if (mode == "loser") start_mode_ = StartPlayerMode::Loser;
  else SpielFatalError("Unknown start mode: " + mode);
  */
}

std::unique_ptr<State> PresidentGame::NewInitialState() const {
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, num_players_ - 1);
  return std::make_unique<PresidentGameState>(shared_from_this(), shuffle_cards_, dist(rng));
}

int PresidentGame::NumPlayers() const { return num_players_; }

int PresidentGame::NumDistinctActions() const { return 1 + kMaxComboSize * kNumRanks; }

double PresidentGame::MinUtility() const { return 0.0; }

double PresidentGame::MaxUtility() const { return num_players_ - 1; }

int PresidentGame::MaxGameLength() const {
  return kNumRanks * num_suits_ * num_players_;
}



// === PresidentGameState ===
PresidentGameState::PresidentGameState(std::shared_ptr<const Game> game_arg, bool shuffle, int start_player)
: State(game_arg), top_rank_(-1), current_combo_size_(0), current_player_(start_player), last_player_to_play_(start_player) {
  
  auto game = static_pointer_cast<const PresidentGame>(game_);
  hands_ = std::vector<Hand>(num_players_, Hand(game->kNumRanks, 0));

  std::vector<int> deck;
  for (int r = 0; r < game->kNumRanks; ++r)
    for (int s = 0; s < game->num_suits_; ++s)
      deck.push_back(r);

  if (shuffle) {
    std::shuffle(deck.begin(), deck.end(), std::mt19937(std::random_device{}()));
  }

  for (int i = 0; i < deck.size(); ++i) {
    hands_[i % num_players_][deck[i]]++;
  }
/*
  switch (game->start_mode_) {
    case PresidentGame::StartPlayerMode::Fixed: current_player_ = 0; break;
    case PresidentGame::StartPlayerMode::Random: {
      std::mt19937 rng(std::random_device{}());
      std::uniform_int_distribution<int> dist(0, num_players_ - 1);
      current_player_ = dist(rng);
      break;
    }
    case PresidentGame::StartPlayerMode::Rotate: current_player_ = rotate_start_index_; break;
    case PresidentGame::StartPlayerMode::Loser: current_player_ = game->last_loser_.value_or(0); break;
  }
  last_player_to_play_ = current_player_;
*/
}

Player PresidentGameState::CurrentPlayer() const { return current_player_; }

std::vector<Action> PresidentGameState::LegalActions() const {
  auto game = static_pointer_cast<const PresidentGame>(game_);
  if (IsOut(current_player_)) {
    return {};
  }
  std::vector<Action> actions;
  bool new_trick = current_player_ == last_player_to_play_;
  if (!new_trick) actions.push_back(0);  // Pass

  int max_combo_size = game->single_card_mode_ ? 1 : game->num_suits_;

  for (int rank = 0; rank < hands_[current_player_].size(); ++rank) {
    int count = hands_[current_player_][rank];
    for (int size = 1; size <= std::min(count, max_combo_size); ++size) {
      if (new_trick || (size == current_combo_size_ && rank > top_rank_)) {
        actions.push_back(EncodeAction({ComboType::Play, size, rank}, game->kNumRanks));
      }
    }
  }

  return actions;
}

void PresidentGameState::ApplyAction(Action action_id) {
  auto game = static_pointer_cast<const PresidentGame>(game_);
  PresidentAction action = DecodeAction(action_id, game->kNumRanks);
  if (action.type == ComboType::Play) {
    ApplyPresidentAction(action, hands_[current_player_]);
    if (IsOut(current_player_)) {
      finish_order_.push_back(current_player_);
    }
    last_player_to_play_ = current_player_;
    top_rank_ = action.rank;
    current_combo_size_ = action.combo_size;
  }

  if (!IsTerminal()) {
    // advance player
    while(true) {
      current_player_ = (current_player_ + 1) % num_players_;
      if (!IsOut(current_player_)) break;
      if (last_player_to_play_ == current_player_) {
        // Player finished with a trick that nobody can play on, next player can start new trick
        last_player_to_play_ = (last_player_to_play_ + 1) % num_players_;
      }
    }
  }
}

bool PresidentGameState::IsTerminal() const {
  return finish_order_.size() == num_players_;
}

std::vector<double> PresidentGameState::Returns() const {
  std::vector<double> scores(num_players_, 0.0);
  for (int i = 0; i < num_players_; i++) {
    scores[finish_order_[i]] = num_players_ - 1 - i;
  }

  /*
  auto game = static_pointer_cast<const PresidentGame>(game_);
  if (game->start_mode_ == PresidentGame::StartPlayerMode::Loser && finish_order_.size() == num_players_ - 1) {
    for (int p = 0; p < num_players_; ++p) {
      if (!IsOut(p)) game->last_loser_ = p;
    }
  }
  if (game->start_mode_ == PresidentGame::StartPlayerMode::Rotate) {
    game->rotate_index_ = (rotate_start_index_ + 1) % num_players_;
  }
  */

  return scores;
}

std::unique_ptr<State> PresidentGameState::Clone() const {
  return std::make_unique<PresidentGameState>(*this);
}

std::string PresidentGameState::ToString() const {
  auto game = static_pointer_cast<const PresidentGame>(game_);
  std::string out = absl::StrFormat("Player %d to play\n", current_player_);
  for (int i = 0; i < num_players_; ++i) {
    absl::StrAppend(&out, "P", i, ": ", HandToString(hands_[i], game->ranks_));
    if (IsOut(i)) absl::StrAppend(&out, "(done)");
    absl::StrAppend(&out, "\n");
  }
  if (top_rank_ >= 0) {
    absl::StrAppend(&out, "Current trick: ", RankToString(top_rank_, game->ranks_), " x", current_combo_size_);
    if (current_player_ == last_player_to_play_) absl::StrAppend(&out, " (new)");
    absl::StrAppend(&out, " (played by P", last_player_to_play_, ")\n");
  }
  if (IsTerminal()) {
    absl::StrAppend(&out, "Finish order: ");
    absl::StrAppend(&out, finish_order_[0]);
    for (int i=1; i < finish_order_.size(); i++) {
      absl::StrAppend(&out, ", ");
      absl::StrAppend(&out, finish_order_[i]); 
    }
    absl::StrAppend(&out, "\n");
  }
  return out;
}

std::string PresidentGameState::ActionToString(Player, Action action_id) const {
  auto game = static_pointer_cast<const PresidentGame>(game_);
  PresidentAction action = DecodeAction(action_id, game->kNumRanks);
  if (action.type == ComboType::Pass) return "Pass";
  return absl::StrFormat("Play %s of %s",
                         ComboToString(action.combo_size),
                         RankToString(action.rank, game->ranks_));
}


std::vector<int> PresidentGameState::GetFinishOrder() {
  return finish_order_;
}

bool PresidentGameState::IsOut(int player) const {
  return std::accumulate(hands_[player].begin(), hands_[player].end(), 0) == 0;
}


// === Für Reinforcement Learning ===
std::vector<int> PresidentGame::ObservationTensorShape() const {
  return {kNumRanks + num_players_ - 1 + 3};
}

void PresidentGameState::ObservationTensor(Player player, absl::Span<float> values) const {
  auto game = static_pointer_cast<const PresidentGame>(game_);
  int last_index = 0;
  for (int i = 0; i < game->kNumRanks; i++) {
    values[i] = static_cast<float>(hands_[player][i]);
  }
  last_index += game->kNumRanks;
  for (int i = 0; i < num_players_ - 1; i++) {
    int other_player = (current_player_ + 1 + i) % num_players_;
    int sum = 0;
    for (int r = 0; r < game->kNumRanks; r++) sum += hands_[other_player][r];
    values[last_index + i] = sum;
  }
  last_index += num_players_ - 1;

  // Siehe ObservationString
  int last_played_relative = ((current_player_ - last_player_to_play_) + num_players_) % num_players_;
  values[last_index] = last_played_relative;
  last_index += 1;
  values[last_index] = last_played_relative == 0 ? 0 : current_combo_size_;
  last_index += 1;
  values[last_index] = last_played_relative == 0 ? -1 : top_rank_;
}

std::string PresidentGameState::ObservationString(Player player) const {
  auto game = std::static_pointer_cast<const PresidentGame>(game_);
  std::string obs;

  // 1. Eigene Handkarten
  for (int i = 0; i < game->kNumRanks; ++i) {
    obs += std::to_string(hands_[player][i]) + ",";
  }

  // 2. Gegnerkartenanzahlen
  for (int i = 0; i < num_players_ - 1; ++i) {
    int other_player = (current_player_ + 1 + i) % num_players_;
    int sum = 0;
    for (int r = 0; r < game->kNumRanks; ++r) sum += hands_[other_player][r];
    obs += std::to_string(sum) + ",";
  }

  // 3. last_played_relative (Vorwärtszählen im Uhrzeigersinn bis zum letzten Ausspieler) <-- Originale Logik
  //int last_played_relative = (last_player_to_play_ - current_player_) % num_players_;
  //obs += std::to_string(last_played_relative) + ",";

  // 3. last_played_relative (Vorwärtszählen im Uhrzeigersinn bis zum letzten Ausspieler, normalisiert)
  //int last_played_relative = ((last_player_to_play_ - current_player_) + num_players_) % num_players_;
  //obs += std::to_string(last_played_relative) + ",";

  // 3. last_played_relative (Rückwärtszählen im Uhrzeigersinn bis zum letzten Ausspieler, normalisiert)
  int last_played_relative = ((current_player_ - last_player_to_play_) + num_players_) % num_players_;
  obs += std::to_string(last_played_relative) + ",";

  // 4. current_combo_size
  obs += (last_played_relative == 0 ? "0," : std::to_string(current_combo_size_) + ",");

  // 5. top_rank
  obs += (last_played_relative == 0 ? "-1" : std::to_string(top_rank_));

  return obs;
}


std::vector<int> PresidentGame::InformationStateTensorShape() const {
  // Gleiche Shape wie ObservationTensor
  return ObservationTensorShape();
}

void PresidentGameState::InformationStateTensor(Player player, absl::Span<float> values) const {
  // Gleicher Inhalt wie ObservationTensor
  ObservationTensor(player, values);
}


std::string PresidentGameState::InformationStateString(Player player) const {
  auto game = std::static_pointer_cast<const PresidentGame>(game_);
  std::string info;

  // 1. Eigene Handkarten (Index 0 - kNumRanks-1)
  for (int i = 0; i < game->kNumRanks; ++i) {
    info += std::to_string(hands_[player][i]) + ",";
  }

  // 2. Gegnerkartenanzahlen (Index kNumRanks - kNumRanks+num_players_-2)
  for (int i = 0; i < num_players_ - 1; ++i) {
    int opponent = (current_player_ + 1 + i) % num_players_;
    int sum = 0;
    for (int r = 0; r < game->kNumRanks; ++r) sum += hands_[opponent][r];
    info += std::to_string(sum) + ",";
  }

  // 3. last_played_relative (Index last_index)
  //int last_played_relative = (last_player_to_play_ - current_player_) % num_players_;
  //info += std::to_string(last_played_relative) + ",";

  // 3. Neue Berechnung, siehe InformationStateTensor
  int last_played_relative = ((current_player_ - last_player_to_play_) + num_players_) % num_players_;
  info += std::to_string(last_played_relative) + ",";

  // 4. current_combo_size (Index last_index + 1)
  if (last_played_relative == 0) {
    info += "0,";
  } else {
    info += std::to_string(current_combo_size_) + ",";
  }

  // 5. top_rank (Index last_index + 2)
  if (last_played_relative == 0) {
    info += "-1";
  } else {
    info += std::to_string(top_rank_);
  }

  return info;
}

}  // namespace president
}  // namespace open_spiel
