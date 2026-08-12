import numpy as np

from base import BanditAgent, LLMAgent

SYSTEM_PROMPT = (
    "You are a bandit algorithm with {0} {unit}s labeled {1}.\n"
    "Each {unit} is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the {unit}s could be different.\n"
    "When you press one of the {unit}s, you will get a reward that is sampled from the {unit}'s associated distribution. Your goal is to maximize the total reward.\n"
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


class UCBMABAgent(BanditAgent):
    def select_arm(self, trial, timestep):
        counts = self.arm_counts[trial]
        # select unused arm first
        if np.any(counts == 0):
            return int(np.argmax(counts == 0))

        bonus = np.sqrt(2 * np.log(timestep + 1) / counts)
        return int(np.argmax(self.arm_rewards[trial] / counts + bonus))


class LLMMABAgent(LLMAgent):
    """LLM bandit agent served by vLLM.

    Overrides `select_arm_batched` because batching is the whole point here: one
    vLLM call per timestep instead of one per (trial, timestep).
    """

    def __init__(self, env, model, seed=0, unit="arm", max_model_len=None):
        super().__init__(env, model, seed=seed, max_model_len=max_model_len)

        self.unit = unit

    def update(self, trial, timestep, arm, reward):
        super().update(trial, timestep, arm, reward)
        self._record_response(trial)

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
                    summary += f"{self.unit.capitalize()} {arm + 1} has not been pressed yet\n"
                    continue
                summary += (
                    f"{self.unit.capitalize()} {arm + 1} was pressed {count:.0f} times "
                    f"with average reward {self.arm_rewards[trial, arm] / count:.2f}\n"
                )

        query = QUERY_TEMPLATE.format(
            unit=self.unit, choices=f"1 to {self.env.num_arms}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": summary + query},
        ]

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
