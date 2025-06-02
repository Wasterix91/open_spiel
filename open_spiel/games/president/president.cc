#include "open_spiel/games/president/president.h"

#include <numeric>
#include <algorithm>
#include <random>
#include <memory>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace president {

// Erzeugt alle legalen Aktionen basierend auf der aktuellen Hand, Spielmodus und Trick-Situation.
// Wenn ein neuer Trick beginnt, dürfen alle gültigen Kombinationen gespielt werden.
// Andernfalls darf nur überboten werden (gleicher Typ, aber höherer Rang).
std::vector<int> LegalActionsFromHand(const Hand& hand, int top_rank, ComboType current_type, bool new_trick, bool single_card_mode) {
  std::vector<int> actions = {0}; // 0 steht für 'Pass'
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

// Entfernt gespielte Karten aus der Hand entsprechend der gespielten Kombination.
void ApplyPresidentAction(const PresidentAction& action, Hand& hand) {
  if (action.type != ComboType::Pass) {
    int count = static_cast<int>(action.type) - static_cast<int>(ComboType::Single) + 1;
    hand[action.rank] -= count;
  }
}

// Wandelt die Hand in einen menschenlesbaren String um (z. B. "8 9 9 O ")
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

// Konstruktor des Spielzustands: initialisiert Deck, mischt ggf. und verteilt Karten
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
      single_card_mode_(std::static_pointer_cast<const PresidentGame>(game)->single_card_mode_) {

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

// Gibt alle legalen Aktionen für den aktuellen Spieler zurück
std::vector<Action> PresidentGameState::LegalActions() const {
  std::vector<int> raw = LegalActionsFromHand(hands_[current_player_], top_rank_, current_combo_type_, new_trick_, single_card_mode_);
  return std::vector<Action>(raw.begin(), raw.end());
}

// Wandelt eine Aktions-ID in einen lesbaren String um (z. B. "Play Pair of 10")
std::string PresidentGameState::ActionToString(Player, Action action_id) const {
  PresidentAction action = DecodeAction(action_id);
  if (action.type == ComboType::Pass) return "Pass";

  static const std::vector<std::string> ranks = {"7", "8", "9", "10", "U", "O", "K", "A"};
  static const std::vector<std::string> combo_names = {"Single", "Pair", "Triple", "Quad"};
  int combo_index = static_cast<int>(action.type) - static_cast<int>(ComboType::Single);
  return "Play " + combo_names[combo_index] + " of " + ranks[action.rank];
}

// Gibt den gesamten Spielzustand als String zurück (z. B. für Debugging oder GUI)
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

// Gibt die Punktzahl (Reward) für alle Spieler am Spielende zurück
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

// Führt eine Aktion aus und aktualisiert den Spielzustand entsprechend
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
    current_combo_type_ = action.type;
    new_trick_ = false;
    std::fill(passed_.begin(), passed_.end(), false);
  }

  AdvanceToNextPlayer();
}

// Wechselt zum nächsten aktiven Spieler oder startet neuen Trick, wenn alle gepasst haben
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

// Gibt die Beobachtung für einen Spieler zurück 
void PresidentGameState::ObservationTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kNumRanks);
  for (int i = 0; i < kNumRanks; ++i) {
    values[i] = static_cast<float>(hands_[player][i]);
  }
}

// Gibt den Informationszustand für einen Spieler zurück (nur eigene Hand)
void PresidentGameState::InformationStateTensor(Player player, absl::Span<float> values) const {
  SPIEL_CHECK_EQ(values.size(), kNumRanks);
  for (int i = 0; i < kNumRanks; ++i) {
    values[i] = static_cast<float>(hands_[player][i]);
  }
}

// Definition des Spiels inklusive Meta-Informationen für OpenSpiel
const GameType kGameType{
    "president",
    "President",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kGeneralSum,
    GameType::RewardModel::kTerminal,
    6, 2,
    false, false, true, false,
    {{"shuffle_cards", GameParameter(true)},
     {"single_card_mode", GameParameter(true)}}
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::make_shared<PresidentGame>(params);
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Konstruktor der Spielklasse
PresidentGame::PresidentGame(const GameParameters& params)
    : Game(kGameType, params),
      shuffle_cards_(ParameterValue<bool>("shuffle_cards")),
      single_card_mode_(ParameterValue<bool>("single_card_mode")) {}

std::unique_ptr<State> PresidentGame::NewInitialState() const {
  return std::make_unique<PresidentGameState>(shared_from_this(), shuffle_cards_);
}

}  // namespace president
}  // namespace open_spiel
