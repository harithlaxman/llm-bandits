from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm


class MABEnv(ABC):
    def __init__(self, num_arms, horizon, num_trials=20, seed=42):
        self.num_arms = num_arms
        self.horizon = horizon
        self.num_trials = num_trials

        self.rng = np.random.default_rng(seed)

        self.arm_means = self._arm_means()

        # every (trial, timestep) has a reward for each arm, drawn from that cell's mean
        self.reward_draws = self.rng.binomial(n=1, p=self.arm_means)

    @abstractmethod
    def _arm_means(self) -> np.ndarray:
        """Success probabilities, shape (num_trials, horizon, num_arms)."""

    def get_reward(self, trial, timestep, arm):
        return self.reward_draws[trial, timestep, arm]

    def get_rewards_batched(self, timestep, arms):
        return self.reward_draws[np.arange(self.num_trials), timestep, arms]

    def run_experiment(self, agent):
        agent.reset()

        for trial in range(self.num_trials):
            for timestep in range(self.horizon):
                arm = agent.select_arm(trial, timestep)
                reward = self.get_reward(trial, timestep, arm)
                agent.update(trial, timestep, arm, reward)

    def run_experiment_batched(self, agent):
        # batch - 1 timestep x num_trials
        agent.reset()

        for timestep in tqdm(range(self.horizon)):
            arms = agent.select_arm_batched(timestep)
            rewards = self.get_rewards_batched(timestep, arms)
            agent.update_batched(timestep, arms, rewards)


class MABStationaryEnv(MABEnv):
    def _arm_means(self):
        # optimal arm redrawn per trial
        self.optimal_arms = self.rng.integers(self.num_arms, size=self.num_trials)
        means = np.full((self.num_trials, self.num_arms), 0.4)
        means[np.arange(self.num_trials), self.optimal_arms] = 0.6

        return np.broadcast_to(
            means[:, None, :], (self.num_trials, self.horizon, self.num_arms)
        )
