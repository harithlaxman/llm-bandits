from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm


class MABEnv(ABC):
    # an env whose rewards need different wording overrides this; None takes the
    # LLM agent's own default
    system_prompt = None

    def __init__(self, num_arms, horizon, num_trials=20, seed=42):
        self.num_arms = num_arms
        self.horizon = horizon
        self.num_trials = num_trials

        self.seed = seed
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

        agent.summary()

    def run_experiment_batched(self, agent):
        # batch - 1 timestep x num_trials
        agent.reset()

        for timestep in tqdm(range(self.horizon)):
            arms = agent.select_arm_batched(timestep)
            rewards = self.get_rewards_batched(timestep, arms)
            agent.update_batched(timestep, arms, rewards)

        agent.summary()


class MABStationaryEnv(MABEnv):
    def _arm_means(self):
        # optimal arm redrawn per trial
        self.optimal_arms = self.rng.integers(self.num_arms, size=self.num_trials)
        means = np.full((self.num_trials, self.num_arms), 0.4)
        means[np.arange(self.num_trials), self.optimal_arms] = 0.6

        return np.broadcast_to(
            means[:, None, :], (self.num_trials, self.horizon, self.num_arms)
        )


class MABNonStationaryEnv(MABEnv):
    """Stationary except at the halfway mark, where the optimal arm shifts by one."""

    # same placeholders as the agent's default, but no promise of a fixed mean;
    # the change point itself is withheld, so finding it is still the agent's job
    system_prompt = (
        "You are a bandit algorithm with {0} {unit}s labeled {1}.\n"
        "Each {unit} is associated with a Bernoulli distribution with an unknown mean that can change over time; the means for the {unit}s could be different.\n"
        "When you press one of the {unit}s, you will get a reward that is sampled from the {unit}'s associated distribution. Your goal is to maximize the total reward.\n"
        "A good strategy to optimize for reward in these situations requires balancing exploration "
        "and exploitation, and it has to keep watching for change. Recent rewards count for more "
        "than old ones, so the best {unit} so far may no longer be the best, and any {unit} that "
        "paid poorly earlier is worth trying again."
    )

    def _arm_means(self):
        # same per-trial draw as the stationary env, so the two are comparable
        first = self.rng.integers(self.num_arms, size=self.num_trials)
        self.switch = self.horizon // 2
        # (num_trials, horizon) - the optimal arm at every round
        self.optimal_arms = np.where(
            np.arange(self.horizon) < self.switch,
            first[:, None],
            ((first + 1) % self.num_arms)[:, None],
        )

        # a real array, not the stationary env's read-only broadcast view
        means = np.full((self.num_trials, self.horizon, self.num_arms), 0.4)
        np.put_along_axis(means, self.optimal_arms[:, :, None], 0.6, axis=2)

        return means
