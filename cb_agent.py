import numpy as np

from base import BanditAgent, LLMAgent

CB_SYSTEM_PROMPT = (
    "You are a contextual bandit algorithm that names the forest cover type of a site.\n"
    "Each round you are shown a description of one site, and you choose one of {0} cover types:\n"
    "{1}\n"
    "You get a reward of 1 if your choice is the true cover type of that site, and 0 if it is not.\n"
    "You are told the reward for your own choice only; you are never told the true cover type.\n"
    "A good strategy to optimize for reward in these situations requires balancing exploration "
    "and exploitation. You need to explore to learn which descriptions go with which cover "
    "types, but you also have to exploit the information that you have to accumulate rewards."
)

# no round count here: it would change every step and break the cached prefix
CB_HISTORY_PREAMBLE = "Here are your past rounds, oldest first:\n\n"

CB_ROUND_TEMPLATE = "Site: {site}\n  -> you chose {name}, reward {reward:.0f}\n\n"

CB_SITE_TEMPLATE = "This is the site you must name now.\nSite: {site}"

CB_QUERY_TEMPLATE = (
    "\n\nWhich cover type will you choose for this site? Respond only with your choice in "
    "the format [choice:<cover_type_number>], where <cover_type_number> is {choices}. "
    "For example: [choice:1]. Do not give an explanation."
)


class CBAgent(BanditAgent):
    """`BanditAgent` bookkeeping with a context threaded through selection.

    The context never reaches the counts or the history - it is recoverable from
    the env's stream, and storing it per timestep would dwarf everything else in
    the log - so `update` drops it and delegates.
    """

    def select_arm(self, trial, timestep, context):
        raise NotImplementedError

    def update(self, trial, timestep, arm, reward, context):
        super().update(trial, timestep, arm, reward)

    # batched over trials
    def select_arm_batched(self, timestep, contexts):
        return np.array([
            self.select_arm(trial, timestep, contexts[trial])
            for trial in range(self.env.num_trials)
        ])

    def update_batched(self, timestep, arms, rewards, contexts):
        for trial, (arm, reward) in enumerate(zip(arms, rewards)):
            self.update(trial, timestep, arm, reward, contexts[trial])


class LinUCBAgent(CBAgent):
    """LinUCB with disjoint linear models - one ridge regression per arm.

    Every arm sees the same context, so an arm's score is
    `theta_a . x + alpha * sqrt(x . A_a^-1 x)`: a ridge estimate plus a bonus that
    shrinks as that arm accumulates evidence along the direction of `x`.

    Only `A_inv` is stored, never `A`. The update is rank-1, so Sherman-Morrison
    gives the new inverse in O(d^2) rather than re-inverting at O(d^3).
    """

    def __init__(self, env, alpha=1.96):
        super().__init__(env)
        self.alpha = alpha

        # every trial is an independent replicate, so each keeps its own models
        shape = (env.num_trials, env.num_arms, env.context_dim)
        self.A_inv = np.empty(shape + (env.context_dim,))
        self.b = np.empty(shape)
        self.theta = np.empty(shape)
        self.reset()

    def reset(self):
        super().reset()
        # ridge prior A = I: the first pull of every arm carries the same bonus,
        # so the opening rounds sweep the arms rather than favouring one
        self.A_inv[:] = np.eye(self.env.context_dim)
        self.b.fill(0.0)
        self.theta.fill(0.0)

    def _run_metadata(self):
        return {**super()._run_metadata(), "alpha": self.alpha}

    def select_arm(self, trial, timestep, context):
        estimate = self.theta[trial] @ context
        # x . A_a^-1 x for every arm at once, no Python loop over arms
        bonus = np.einsum('d,kde,e->k', context, self.A_inv[trial], context)
        # A_inv stays positive definite, so a negative here is rounding only
        return int(np.argmax(estimate + self.alpha * np.sqrt(np.maximum(bonus, 0.0))))

    def update(self, trial, timestep, arm, reward, context):
        super().update(trial, timestep, arm, reward, context)
        # Sherman-Morrison for A_a += x x^T
        scaled = self.A_inv[trial, arm] @ context
        self.A_inv[trial, arm] -= np.outer(scaled, scaled) / (1.0 + context @ scaled)
        self.b[trial, arm] += reward * context
        self.theta[trial, arm] = self.A_inv[trial, arm] @ self.b[trial, arm]


class LLMCBAgent(LLMAgent, CBAgent):
    """LLM contextual bandit agent served by vLLM, reading the env's text view.

    The prompt is system, then history, then the current site, so the part that
    grows stays a prefix and vLLM caches it between timesteps. `history_window`
    caps the rounds shown; `None` shows every round so far.
    """

    def __init__(self, env, model, seed=0, history_window=None, max_model_len=None):
        self.history_window = history_window

        # checked before the model loads, so a prompt that cannot fit fails at once
        estimate = self._estimate_tokens(env)
        print(f"worst-case prompt is about {estimate} tokens")
        if max_model_len is not None and estimate > max_model_len:
            raise ValueError(
                f"worst-case prompt of about {estimate} tokens exceeds max_model_len "
                f"{max_model_len}; lower history_window (now {history_window}) or "
                f"raise max_model_len"
            )

        super().__init__(env, model, seed=seed, max_model_len=max_model_len)
        # `CBAgent.update` drops the context, so the rendered sites are kept here
        self.context_texts = [[] for _ in range(env.num_trials)]

    def reset(self):
        super().reset()
        self.context_texts = [[] for _ in range(self.env.num_trials)]

    def _system_prompt(self, env):
        names = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(env.arm_names))
        return CB_SYSTEM_PROMPT.format(env.num_arms, names)

    def _run_metadata(self):
        return {**super()._run_metadata(), "history_window": self.history_window}

    def _history_record(self, trial, index):
        # the rendered site is kept alongside the history, so merge it back in
        record = super()._history_record(trial, index)
        record["context_text"] = self.context_texts[trial][index]

        return record

    def _estimate_tokens(self, env):
        # one rendered round, times the rounds shown, at roughly 4 chars per token
        site = env.render_context(env.get_context(0, 0))
        example = CB_ROUND_TEMPLATE.format(
            site=site, name=max(env.arm_names, key=len), reward=0
        )
        fixed = len(self._system_prompt(env)) + len(site) + len(CB_QUERY_TEMPLATE)
        rounds = self.history_window or env.horizon

        return (fixed + rounds * len(example)) // 4

    # shared core
    def _build_prompt(self, trial, timestep, site):
        # the system message is identical across trials, so it is a shared prefix
        history = ""
        if timestep > 0:
            entries, texts = self.history[trial], self.context_texts[trial]
            if self.history_window is not None:
                entries = entries[-self.history_window:]
                texts = texts[-self.history_window:]
            history = CB_HISTORY_PREAMBLE + "".join(
                CB_ROUND_TEMPLATE.format(
                    site=text,
                    name=self.env.arm_names[entry["chosen_arm"]],
                    reward=entry["reward"],
                )
                for text, entry in zip(texts, entries)
            )

        query = CB_QUERY_TEMPLATE.format(choices=f"1 to {self.env.num_arms}")
        return [
            {"role": "system", "content": self._system_prompt(self.env)},
            {
                "role": "user",
                "content": history + CB_SITE_TEMPLATE.format(site=site) + query,
            },
        ]

    def select_arm(self, trial, timestep, context):
        outputs = self.model.chat(
            [self._build_prompt(trial, timestep, self.env.render_context(context))],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return self._parse_arm(trial, outputs[0])

    def update(self, trial, timestep, arm, reward, context):
        super().update(trial, timestep, arm, reward, context)
        self._record_response(trial)
        self.context_texts[trial].append(self.env.render_context(context))

    # batched over trials
    def select_arm_batched(self, timestep, contexts):
        # outputs come back in request order, so output i belongs to trial i
        trials = range(self.env.num_trials)
        sites = [self.env.render_context(context) for context in contexts]
        outputs = self.model.chat(
            [self._build_prompt(trial, timestep, sites[trial]) for trial in trials],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return np.array([
            self._parse_arm(trial, output) for trial, output in zip(trials, outputs)
        ])
