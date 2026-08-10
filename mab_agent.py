import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from base import BanditAgent

SYSTEM_PROMPT = (
    "You are a bandit algorithm with {0} buttons labeled {1}.\n"
    "Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the buttons could be different.\n"
    "When you press a button, you will get a reward that is sampled from the button's associated distribution. Your goal is to maximize the total reward.\n"
    "A good strategy to optimize for reward in these situations requires balancing exploration "
    "and exploitation. You need to explore to try out all of the buttons and find those with high "
    "rewards, but you also have to exploit the information that you have to "
    "accumulate rewards."
)

SUMMARY_HISTORY_PREAMBLE = "So far you have interacted {n} times. Here is a summary of your choices and rewards for each button:\n"

QUERY_TEMPLATE = "\n\nWhich {unit} will you choose next? PLEASE RESPOND ONLY WITH {choices} AND NO TEXT EXPLANATION."


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

    def __init__(self, env, model, seed=0):
        super().__init__(env)

        self.seed = seed
        self.model = LLM(
            model,
            enable_prefix_caching=True,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=20,
            structured_outputs=StructuredOutputsParams(choice=self._choices()),
        )

    def _choices(self):
        return [str(i + 1) for i in range(self.env.num_arms)]

    def _sampling_params(self, trial, timestep):
        params = self.sampling_params.clone()
        params.seed = self.seed + trial * self.env.horizon + timestep
        return params

    # shared core
    def _build_prompt(self, trial, timestep):
        # one chat conversation from one trial's per-arm stats; the system
        # message is identical across trials so it is a cached shared prefix
        system = SYSTEM_PROMPT.format(
            self.env.num_arms,
            ", ".join(str(i + 1) for i in range(self.env.num_arms)),
        )

        summary = ""
        if timestep > 0:
            summary = SUMMARY_HISTORY_PREAMBLE.format(n=timestep)
            for arm in range(self.env.num_arms):
                count = self.arm_counts[trial, arm]
                if not count:
                    # never claim an average for an unpulled arm - "average
                    # reward 0.00" reads as evidence against exploring it
                    summary += f"Button {arm + 1} has not been pressed yet\n"
                    continue
                summary += (
                    f"Button {arm + 1} was pressed {count:.0f} times "
                    f"with average reward {self.arm_rewards[trial, arm] / count:.2f}\n"
                )

        query = QUERY_TEMPLATE.format(
            unit="button", choices=f"1 to {self.env.num_arms}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": summary + query},
        ]

    def _parse_arm(self, output):
        choices = self._choices()
        if not output.outputs:
            # no completion at all - a preempted/aborted request, not a bad parse
            raise RuntimeError(
                f"vLLM returned no completions; expected one of {choices}"
            )

        # match the label exactly rather than int()-ing it: int() also accepts
        # "+3", "03" and unicode digits, none of which the grammar can emit
        text = output.outputs[0].text.strip()
        if text not in choices:
            raise ValueError(f"generated text {text!r} is not one of {choices}")
        return int(text) - 1

    def select_arm(self, trial, timestep):
        outputs = self.model.chat(
            [self._build_prompt(trial, timestep)],
            sampling_params=self._sampling_params(trial, timestep),
        )
        return self._parse_arm(outputs[0])

    # batched over trials
    def select_arm_batched(self, timestep):
        # outputs come back in request order, so output i belongs to trial i;
        # the `SamplingParams` list is paired positionally with the prompts, so
        # each trial keeps its own seed regardless of what else shares the batch
        trials = range(self.env.num_trials)
        outputs = self.model.chat(
            [self._build_prompt(trial, timestep) for trial in trials],
            sampling_params=[
                self._sampling_params(trial, timestep) for trial in trials
            ],
        )
        return np.array([self._parse_arm(output) for output in outputs])
