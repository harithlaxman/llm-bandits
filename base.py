import numpy as np
import matplotlib.pyplot as plt


class BanditAgent:
    """Bookkeeping shared by every agent, in matched sequential/`_batched` pairs."""

    def __init__(self, env):
        self.env = env
        # NaN rather than 0: rewards are Bernoulli, so a leftover 0 from a run
        # that died partway would pass for a genuine zero reward
        self.rewards = np.full((env.num_trials, env.horizon), np.nan)
        self.arm_counts = np.zeros((env.num_trials, env.num_arms))
        self.arm_rewards = np.zeros((env.num_trials, env.num_arms))
        # one list per trial of {"timestep", "chosen_arm", "reward"} dicts
        self.history = [[] for _ in range(env.num_trials)]

    def reset(self):
        self.rewards.fill(np.nan)
        self.arm_counts.fill(0)
        self.arm_rewards.fill(0)
        self.history = [[] for _ in range(self.env.num_trials)]

    def select_arm(self, trial, timestep):
        raise NotImplementedError

    def select_arm_batched(self, timestep):
        return np.array([
            self.select_arm(trial, timestep) for trial in range(self.env.num_trials)
        ])

    def update(self, trial, timestep, arm, reward):
        self.rewards[trial, timestep] = reward
        self.arm_counts[trial, arm] += 1
        self.arm_rewards[trial, arm] += reward
        self.history[trial].append({
            "timestep": timestep,
            "chosen_arm": int(arm),
            "reward": float(reward),
        })

    def update_batched(self, timestep, arms, rewards):
        for trial, (arm, reward) in enumerate(zip(arms, rewards)):
            self.update(trial, timestep, arm, reward)

    def cumulative_regrets(self):
        """Pseudo-regret per trial, cumulative over time, shape (num_trials, horizon)."""
        cumulative_regrets = []

        for trial in range(self.env.num_trials):
            timesteps = np.array([
                entry["timestep"] for entry in self.history[trial]
            ], dtype=int)
            chosen_arms = np.array([
                entry["chosen_arm"] for entry in self.history[trial]
            ], dtype=int)
            chosen_arm_means = self.env.arm_means[trial, timesteps, chosen_arms]
            optimal_arm_means = np.max(
                self.env.arm_means[trial, timesteps, :], axis=1
            )
            cumulative_regrets.append(
                np.cumsum(optimal_arm_means - chosen_arm_means)
            )

        return np.stack(cumulative_regrets)

    def plot_regret(self):
        mean_cumulative_regret = np.mean(self.cumulative_regrets(), axis=0)
        # per-timestep regret averages the same curve over elapsed steps, so it
        # decays towards 0 once the agent settles on the optimal arm
        steps = np.arange(1, mean_cumulative_regret.size + 1)

        fig, (cumulative_ax, per_timestep_ax) = plt.subplots(1, 2, figsize=(10, 4))
        cumulative_ax.plot(mean_cumulative_regret)
        cumulative_ax.set_xlabel("Timestep")
        cumulative_ax.set_ylabel("Mean cumulative regret")

        per_timestep_ax.plot(mean_cumulative_regret / steps)
        per_timestep_ax.set_xlabel("Timestep")
        per_timestep_ax.set_ylabel("Mean per-timestep regret")

        fig.tight_layout()
        plt.show()
