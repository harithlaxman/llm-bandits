import re

import numpy as np
import matplotlib.pyplot as plt

CHOICE_PATTERN = re.compile(r"\[choice:\s*(\d+)\s*\]")


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

    def optimal_arm_choices(self):
        """1.0 where the chosen arm was optimal, shape (num_trials, horizon)."""
        optimal_arm_choices = []

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
            # compare means, not indices, so tied optimal arms both count
            optimal_arm_choices.append(
                (chosen_arm_means == optimal_arm_means).astype(float)
            )

        return np.stack(optimal_arm_choices)

    def summary(self):
        """Print end-of-run stats, each a mean +/- std over trials."""
        # nansum, so a run that died partway still summarises
        rewards = np.nansum(self.rewards, axis=1)
        print(f"cumulative reward = {rewards.mean():.2f} +/- {rewards.std():.2f}")

        # regret and the optimal-arm rate need arm means, which contextual envs lack
        if not hasattr(self.env, "arm_means"):
            return

        regrets = self.cumulative_regrets()[:, -1]
        # second half only: a whole-run rate is dragged down by early exploration
        suffix = self.optimal_arm_choices()[:, self.env.horizon // 2:].mean(axis=1) * 100
        print(f"cumulative regret = {regrets.mean():.2f} +/- {regrets.std():.2f}")
        print(f"optimal arm (2nd half) = {suffix.mean():.1f}% +/- {suffix.std():.1f}%")

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

    def plot_ctr(self):
        steps = np.arange(1, self.env.horizon + 1)
        curves = {"Reward": self.rewards}
        # the optimal-arm rate needs arm means, which contextual envs lack
        if hasattr(self.env, "arm_means"):
            curves["Optimal arm"] = self.optimal_arm_choices()

        fig, running_ax = plt.subplots(figsize=(5, 4))
        for label, picks in curves.items():
            running_ax.plot(np.mean(np.cumsum(picks, axis=1) / steps, axis=0), label=label)

        running_ax.set_xlabel("Timestep")
        running_ax.set_ylabel("Mean running CTR")
        running_ax.set_ylim(0, 1)
        running_ax.legend()

        fig.tight_layout()
        plt.show()


class LLMAgent(BanditAgent):
    """vLLM plumbing shared by the MAB and contextual LLM agents.

    Adds the model and the parsing to the usual bookkeeping; subclasses supply the
    prompt. A contextual subclass lists `CBAgent` after this one, which puts it
    between here and `BanditAgent`, so the contextual `update` still runs.
    """

    def __init__(self, env, model, seed=0, max_model_len=None):
        # imported here, not at module level: a run without an LLM agent should
        # not pay for the vLLM and CUDA import
        from vllm import LLM, SamplingParams

        super().__init__(env)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = LLM(
            model,
            enable_prefix_caching=True,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )

        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_tokens=64,
        )

        # generation is unconstrained, so a response can miss the tag
        self.parse_failures = np.zeros(env.num_trials, dtype=int)
        self.last_responses = [None] * env.num_trials

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self.parse_failures.fill(0)
        self.last_responses = [None] * self.env.num_trials

    def _parse_arm(self, trial, output):
        if not output.outputs:
            # no completion at all - a preempted/aborted request, not a bad parse
            raise RuntimeError("vLLM returned no completions")

        text = output.outputs[0].text
        self.last_responses[trial] = text

        # last match wins: a model that restates itself ends on its final answer
        matches = CHOICE_PATTERN.findall(text.lower())
        if matches:
            arm = int(matches[-1]) - 1
            if 0 <= arm < self.env.num_arms:
                return arm

        self.parse_failures[trial] += 1
        return int(self.rng.integers(self.env.num_arms))

    def _record_response(self, trial):
        # the update signatures differ between MAB and CB, so each agent calls this
        self.history[trial][-1]["raw_response"] = self.last_responses[trial]

    def summary(self):
        super().summary()
        failures = self.parse_failures
        share = failures.mean() / self.env.horizon * 100
        print(
            f"parse failures = {failures.mean():.2f} +/- {failures.std():.2f} "
            f"per trial ({share:.1f}% of pulls, {failures.sum()} total)"
        )
