"""
Minimal PPO-Implementation aus OpenSpiel (angepasst & gekürzt).

Quelle: OpenSpiel python/algorithms/ppo.py
Lizenz: Apache 2.0
"""

import collections
import copy
import numpy as np
import tensorflow as tf

PPOConfig = collections.namedtuple(
    "PPOConfig",
    [
        "learning_rate",
        "num_epochs",
        "batch_size",
        "entropy_cost",
    ]
)

DEFAULT_CONFIG = PPOConfig(
    learning_rate=0.001,
    num_epochs=5,
    batch_size=32,
    entropy_cost=0.01,
)

class PPOAgent:
    def __init__(self, info_state_size, num_actions, config=None):
        self._info_state_size = info_state_size
        self._num_actions = num_actions
        self._config = config or DEFAULT_CONFIG

        # Simple policy & value network
        self._build_networks()
        self._buffer = []

    def _build_networks(self):
        self._policy = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self._info_state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self._num_actions, activation="softmax")
        ])
        self._value = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self._info_state_size,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._config.learning_rate)

    def step(self, time_step):
        if time_step.last():
            # Terminal: nutze finalen Reward für ALLE Schritte
            final_reward = time_step.rewards[0]
            self._buffer = [(s, a, final_reward) for (s, a, _) in self._buffer]
            self._train()
            self._buffer = []
            return None

        info_state = np.array(time_step.observations["info_state"][0])
        probs = self._policy(info_state[None, :]).numpy()[0]
        action = np.random.choice(self._num_actions, p=probs)

        # Buffer: speichere 0 als Reward-Placeholder
        self._buffer.append((info_state, action, 0.0))

        return collections.namedtuple("AgentOutput", ["action"])(action=action)


    def _train(self):
        if not self._buffer:
            return
        states, actions, rewards = zip(*self._buffer)
        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(rewards)

        for _ in range(self._config.num_epochs):
            with tf.GradientTape() as tape:
                logits = self._policy(states)
                values = self._value(states)[:, 0]
                one_hot = tf.one_hot(actions, self._num_actions)
                action_probs = tf.reduce_sum(logits * one_hot, axis=1)

                advantage = returns - values
                policy_loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-10) * advantage)
                value_loss = tf.reduce_mean(tf.square(advantage))
                entropy = -tf.reduce_mean(tf.reduce_sum(logits * tf.math.log(logits + 1e-10), axis=1))
                loss = policy_loss + 0.5 * value_loss - self._config.entropy_cost * entropy

            grads = tape.gradient(loss, self._policy.trainable_variables + self._value.trainable_variables)
            self._optimizer.apply_gradients(
                zip(grads, self._policy.trainable_variables + self._value.trainable_variables)
            )

    def save(self, path):
        self._policy.save(path + "_policy")
        self._value.save(path + "_value")

    def restore(self, path):
        self._policy = tf.keras.models.load_model(path + "_policy")
        self._value = tf.keras.models.load_model(path + "_value")
