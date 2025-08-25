import numpy as np
import json


class ValueTableAgent:
    index_dt = np.dtype(np.uint64)
    data_dt = np.dtype([('next', index_dt), ('v', np.float32),
        ('v_count', np.uint16), ('action', np.uint8), ('_dummy', np.uint8)])
    sentinel_addr = 0xFFFFFFFFFFFFFFFF

    def __init__(self, data_files_path_prefix):
        with open(data_files_path_prefix + "_params.json") as f:
            params_obj = json.load(f)
        self.n_players = params_obj['n_players']
        self.n_ranks = params_obj['n_ranks']
        self.n_cards_per_rank = params_obj['n_cards_per_rank']
        self.index = np.fromfile(data_files_path_prefix + "_index.bin", dtype=ValueTableAgent.index_dt)
        self.data = np.fromfile(data_files_path_prefix + "_data.bin", dtype=ValueTableAgent.data_dt)

    def remove_empty_players(self, player_sums, last_played):
        shifted = []
        last_played_out = last_played
        n_removed = 0
        for i in range(self.n_players - 1):
            if player_sums[i] or last_played == i + 1:
                shifted.append(player_sums[i])
            else:
                n_removed += 1
                if len(shifted) < last_played_out:
                    last_played_out -= 1
        shifted.extend(n_removed * [0])
        return shifted, last_played_out

    def get_action_values(self, state_observation):
        #print(state_observation)
        state_observation = [int(x) for x in state_observation]
        own_hand = state_observation[:self.n_ranks]
        other_hand_sums = list(reversed(state_observation[self.n_ranks:
            self.n_ranks + self.n_players - 1]))
        last_played = state_observation[self.n_ranks + self.n_players - 1]
        trick_count_minus_one = state_observation[self.n_ranks + self.n_players] - 1
        trick_rank = state_observation[self.n_ranks + self.n_players + 1]
        if not last_played:
            trick_rank = 0
            trick_count_minus_one = 0
        cards_gone = state_observation[self.n_ranks + self.n_players + 2:]
        other_hand_sums_shifted, last_played_shifted = self.remove_empty_players(other_hand_sums, last_played)
        state = [last_played_shifted, trick_rank, trick_count_minus_one] + list(own_hand) + \
            other_hand_sums_shifted + list(cards_gone)
        #print(state)

        address = 0
        for x in state:
            if address == ValueTableAgent.sentinel_addr:
                raise IndexError(state)
            address = self.index[address + x]
        action_dict = {}
        while address != ValueTableAgent.sentinel_addr:
            elem = self.data[address]
            action_dict[int(elem['action'])] = float(elem['v'])
            address = elem['next']
        #print(action_dict)
        return action_dict

    def table_code_to_openspiel_code(self, code):
        if code == 0xFF:
            return 0
        else:
            rank, count_minus_one = divmod(code, self.n_cards_per_rank)
            return 1 + count_minus_one * self.n_ranks + rank

    def openspiel_code_to_table_code(self, code):
        if code == 0:
            return 0xFF
        count_minus_one, rank = divmod(code - 1, self.n_ranks)
        return rank * self.n_cards_per_rank + count_minus_one

    def __call__(self, info_state):
        action_dict = self.get_action_values(info_state.observation_tensor())
        #transcoded_actions = set(self.table_code_to_openspiel_code(c) for c in action_dict.keys())
        #assert set(legal_actions) == transcoded_actions, \
        #    f"legal_actions don't match (supplied argument: {set(legal_actions)}, in table: {transcoded_actions})"
        best_action = min(action_dict.items(), key=lambda x: x[1])[0]
        return self.table_code_to_openspiel_code(best_action)
