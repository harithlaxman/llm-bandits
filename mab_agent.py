import re

import numpy as np
from vllm import LLM, SamplingParams

from base import BanditAgent

SYSTEM_PROMPT = (
    "You are a bandit algorithm with {0} {unit}s labeled {1}.\n"
    "Each {unit} is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the {unit}s could be different.\n"
    "When you press a {unit}, you will get a reward that is sampled from the {unit}'s associated distribution. Your goal is to maximize the total reward.\n"
    "A good strategy to optimize for reward in these situations requires balancing exploration "
    "and exploitation. You need to explore to try out all of the {unit}s and find those with high "
    "rewards, but you also have to exploit the information that you have to "
    "accumulate rewards."
)

SUMMARY_HISTORY_PREAMBLE = "So far you have interacted {n} times. Here is a summary of your choices and rewards for each {unit}:\n"

QUERY_TEMPLATE = (
    "\n\nWhich {unit} will you choose next? Respond only with your choice in the "
    "format [choice:<{unit}_number>], where <{unit}_number> is {choices}. "
    "For example: [choice:1]. Do not give an explanation."
)

CHOICE_PATTERN = re.compile(r"\[choice:\s*(\d+)\s*\]")


class UCBMABAgent(BanditAgent):
    def select_arm(self, trial, timestep):
        counts = self.arm_counts[trial]
        # select unused arm first
        if np.any(counts == 0):
            return int(np.argmax(counts == 0))

        bonus = np.sqrt(2 * np.log(timestep + 1) / counts)
        return int(np.argmax(self.arm_rewards[trial] / counts + bonus))


class LLMMABAgent(BanditAgent):
    """LLM bandit agent served by vLLM.

    Overrides `select_arm_batched` because batching is the whole point here: one
    vLLM call per timestep instead of one per (trial, timestep).
    """

    def __init__(self, env, model, seed=0, unit="arm"):
        super().__init__(env)

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.model = LLM(
            model,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            max_tokens=64,
        )

        # generation is unconstrained now, so a response can miss the tag
        self.parse_failures = np.zeros(env.num_trials, dtype=int)
        self.last_responses = [None] * env.num_trials

        self.unit = unit

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng(self.seed)
        self.parse_failures.fill(0)
        self.last_responses = [None] * self.env.num_trials

    def update(self, trial, timestep, arm, reward):
        super().update(trial, timestep, arm, reward)
        self.history[trial][-1]["raw_response"] = self.last_responses[trial]

    def summary(self):
        super().summary()
        failures = self.parse_failures
        share = failures.mean() / self.env.horizon * 100
        print(
            f"parse failures = {failures.mean():.2f} +/- {failures.std():.2f} "
            f"per trial ({share:.1f}% of pulls, {failures.sum()} total)"
        )

    # shared core
    def _build_prompt(self, trial, timestep):
        # one chat conversation from one trial's per-arm stats; the system
        # message is identical across trials so it is a cached shared prefix
        system = SYSTEM_PROMPT.format(
            self.env.num_arms,
            ", ".join(str(i + 1) for i in range(self.env.num_arms)),
            unit=self.unit,
        )

        summary = ""
        if timestep > 0:
            summary = SUMMARY_HISTORY_PREAMBLE.format(n=timestep, unit=self.unit)
            for arm in range(self.env.num_arms):
                count = self.arm_counts[trial, arm]
                if not count:
                    # never claim an average for an unpulled arm - "average
                    # reward 0.00" reads as evidence against exploring it
                    summary += f"{self.unit} {arm + 1} has not been pressed yet\n"
                    continue
                summary += (
                    f"{self.unit} {arm + 1} was pressed {count} times "
                    f"with average reward {self.arm_rewards[trial, arm] / count:.2f}\n"
                )

        query = QUERY_TEMPLATE.format(
            unit=self.unit, choices=f"1 to {self.env.num_arms}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": summary + query},
        ]

    def _parse_arm(self, trial, output):
        if not output.outputs:
            # no completion at all - a preempted/aborted request, not a bad parse
            raise RuntimeError("vLLM returned no completions")

        text = output.outputs[0].text
        self.last_responses[trial] = text

        # last match wins: a model that restates itself ends on its final answer
        matches = CHOICE_PATTERN.findall(text)
        if matches:
            arm = int(matches[-1]) - 1
            if 0 <= arm < self.env.num_arms:
                return arm

        self.parse_failures[trial] += 1
        return int(self.rng.integers(self.env.num_arms))

    def select_arm(self, trial, timestep):
        outputs = self.model.chat(
            [self._build_prompt(trial, timestep)],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return self._parse_arm(trial, outputs[0])

    # batched over trials
    def select_arm_batched(self, timestep):
        # outputs come back in request order, so output i belongs to trial i
        trials = range(self.env.num_trials)
        outputs = self.model.chat(
            [self._build_prompt(trial, timestep) for trial in trials],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return np.array([
            self._parse_arm(trial, output) for trial, output in zip(trials, outputs)
        ])
