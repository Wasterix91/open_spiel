import torch
import deep_cfr

class DeepCFRSolver:
    def __init__(self, game, num_iterations, num_traversals, learning_rate,
                 batch_size, memory_capacity, device="cpu"):
        self.solver = deep_cfr.DeepCFRSolver(
            game=game,
            policy_network_layers=[128, 64],
            advantage_network_layers=[128, 64],
            num_iterations=num_iterations,
            num_traversals=num_traversals,
            learning_rate=learning_rate,
            batch_size=batch_size,
            memory_capacity=memory_capacity,
            policy_network_train_steps=100,
            advantage_network_train_steps=100,
            device=device,
        )

    def solve(self):
        return self.solver.solve()
