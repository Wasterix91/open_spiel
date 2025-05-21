// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/president/president.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace president {
namespace {

namespace testing = open_spiel::testing;

void BasicPresidentTests() {
  testing::LoadGameTest("president");
  testing::ChanceOutcomesTest(*LoadGame("president"));
  testing::RandomSimTest(*LoadGame("president"), 100);
  testing::RandomSimTest(*LoadGame("president",
                         {{"action_mapping", GameParameter(true)}}), 100);
  testing::RandomSimTest(*LoadGame("president",
                         {{"suit_isomorphism", GameParameter(true)}}), 100);
  for (Player players = 3; players <= 5; players++) {
    testing::RandomSimTest(
        *LoadGame("president", {{"players", GameParameter(players)}}), 100);
  }
  testing::ResampleInfostateTest(*LoadGame("president"), /*num_sims=*/100);
  auto observer = LoadGame("president")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("president"), observer);
}

void PolicyTest() {
  using PolicyGenerator = std::function<TabularPolicy(const Game& game)>;
  std::vector<PolicyGenerator> policy_generators = {
      GetAlwaysFoldPolicy,
      GetAlwaysCallPolicy,
      GetAlwaysRaisePolicy
  };

  std::shared_ptr<const Game> game = LoadGame("president");
  for (const auto& policy_generator : policy_generators) {
    testing::TestEveryInfostateInPolicy(policy_generator, *game);
    testing::TestPoliciesCanPlay(policy_generator, *game);
  }
}

}  // namespace
}  // namespace president
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::president::BasicPresidentTests();
  open_spiel::president::PolicyTest();
}
