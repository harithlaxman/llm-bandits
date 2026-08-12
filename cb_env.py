from abc import ABC, abstractmethod

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from tqdm import tqdm


NUM_CONTINUOUS, NUM_BINS, NUM_WILDERNESS = 10, 5, 4

FEATURE_NAMES = [
    "Elevation", "Aspect", "Slope", "Horizontal distance to water",
    "Vertical distance to water", "Horizontal distance to road",
    "Hillshade 9am", "Hillshade noon", "Hillshade 3pm",
    "Horizontal distance to fire point",
]
BIN_LABELS = ["very low", "low", "medium", "high", "very high"]
# the four areas and the seven cover types, both named in the OpenML description
AREA_NAMES = ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
COVER_TYPE_NAMES = {
    "Aspen": "Aspen",
    "Cottonwood_Willow": "Cottonwood/Willow",
    "Douglas_fir": "Douglas-fir",
    "Krummholz": "Krummholz",
    "Lodgepole_Pine": "Lodgepole Pine",
    "Ponderosa_Pine": "Ponderosa Pine",
    "Spruce_Fir": "Spruce/Fir",
}


def _binarize(features):
    """(N, 14) terrain + wilderness -> (N, 54) binary.

    Each terrain column becomes a width-5 one-hot over its own quintiles. The width
    is fixed, so the heavy ties in this dataset leave a bin rare rather than moving
    every column that follows.
    """
    terrain, wilderness = features[:, :NUM_CONTINUOUS], features[:, NUM_CONTINUOUS:]

    blocks = []
    for column in terrain.T:
        edges = np.quantile(column, [0.2, 0.4, 0.6, 0.8])
        bins = np.searchsorted(edges, column, side="right")
        one_hot = np.zeros((len(column), NUM_BINS), dtype=np.uint8)
        one_hot[np.arange(len(column)), bins] = 1
        blocks.append(one_hot)

    return np.hstack(blocks + [wilderness.astype(np.uint8)])


def load_covertype(num_samples=None, seed=42):
    """A class-balanced subsample of Covertype, an equal number of rows per class.

    The 40 soil columns are dropped and the 10 terrain columns are binned, leaving
    54 binary features that a text renderer can decode back into words. Binning
    comes after the balanced draw, so each bin holds about a fifth of the rows an
    agent actually sees.

    `num_samples` is a ceiling on the total: each class contributes
    `num_samples // num_classes` rows, itself capped by the rarest class (1339), so
    the result can fall short of what was asked. `None` takes the rarest-class cap,
    i.e. 9373 rows.
    """
    data = fetch_openml("covertype", version=1, as_frame=False, parser="auto")
    features = data["data"][:, :NUM_CONTINUOUS + NUM_WILDERNESS].astype(float)

    encoder = OrdinalEncoder(dtype=int)
    labels = encoder.fit_transform(data["target"].reshape(-1, 1))
    labels = labels.astype(int).reshape(-1)
    # the encoder sorts the class strings, so arm order is alphabetical, not UCI 1-7
    arm_names = [COVER_TYPE_NAMES[name] for name in encoder.categories_[0]]

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = np.bincount(labels).min()
    if num_samples is not None:
        per_class = min(num_samples // len(classes), per_class)
    if per_class < 1:
        raise ValueError(
            f"num_samples {num_samples} cannot cover {len(classes)} classes; "
            f"a balanced draw needs at least {len(classes)}"
        )

    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == label), size=per_class, replace=False)
        for label in classes
    ])
    # the per-class draw leaves the pool sorted by label; shuffle so anything
    # reading it in order sees a stationary stream rather than class blocks
    rng.shuffle(indices)
    return _binarize(features[indices]), labels[indices], arm_names


class CBEnv(ABC):
    """Everything a contextual bandit experiment needs except the data.

    Mirrors `MABEnv`: a subclass supplies the context and reward lookups, and the
    two drivers here are the only thing an agent is ever driven by. The contextual
    drivers differ from the MAB ones only in fetching a context and passing it on.
    """

    # a contextual env without per-arm means leaves this None
    arm_means = None
    # an env whose rewards need different wording overrides this; None takes the
    # LLM agent's own default
    system_prompt = None

    def __init__(self, num_arms, context_dim, horizon, num_trials=20, seed=42):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.horizon = horizon
        self.num_trials = num_trials

        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def get_context(self, trial, timestep) -> np.ndarray:
        """One trial's context at `timestep`, shape (context_dim,)."""

    @abstractmethod
    def get_contexts_batched(self, timestep) -> np.ndarray:
        """Every trial's context at `timestep`, shape (num_trials, context_dim)."""

    @abstractmethod
    def get_reward(self, trial, timestep, arm) -> float:
        """Reward for pulling `arm` on one trial's context at `timestep`."""

    @abstractmethod
    def get_rewards_batched(self, timestep, arms) -> np.ndarray:
        """Reward for each trial's own pull at `timestep`, shape (num_trials,)."""

    def run_experiment(self, agent):
        agent.reset()

        for trial in range(self.num_trials):
            for timestep in range(self.horizon):
                context = self.get_context(trial, timestep)
                arm = agent.select_arm(trial, timestep, context)
                reward = self.get_reward(trial, timestep, arm)
                agent.update(trial, timestep, arm, reward, context)

        agent.summary()

    def run_experiment_batched(self, agent):
        # every trial advances in lockstep, so a whole timestep is one set of
        # array ops rather than `num_trials` separate ones
        agent.reset()

        for timestep in tqdm(range(self.horizon)):
            contexts = self.get_contexts_batched(timestep)
            arms = agent.select_arm_batched(timestep, contexts)
            rewards = self.get_rewards_batched(timestep, arms)
            agent.update_batched(timestep, arms, rewards, contexts)

        agent.summary()


class ClassificationCBEnv(CBEnv):
    """A labelled dataset as a contextual bandit: name the class, score 1 if right.

    Arms are the classes and a context is one row of features, so any
    `(features, labels)` pair works - `load_covertype` is one such loader.

    Each trial draws its own `horizon` rows from the pool, without replacement
    within a trial but freely overlapping across trials. Trials are therefore
    independent samples, so their spread is sampling variance rather than the
    ordering variance a shared pool would give.
    """

    def __init__(self, features, labels, horizon, num_trials=20, seed=42):
        if not 1 <= horizon <= len(features):
            raise ValueError(
                f"horizon {horizon} must be between 1 and the pool of "
                f"{len(features)} samples"
            )

        super().__init__(
            num_arms=len(np.unique(labels)),
            context_dim=features.shape[1],
            horizon=horizon,
            num_trials=num_trials,
            seed=seed,
        )
        self.features = features
        self.labels = labels

        # (num_trials, horizon) indices into the pool, an independent draw per trial
        self.stream = np.array([
            self.rng.choice(len(features), size=horizon, replace=False)
            for _ in range(num_trials)
        ])
        # (num_trials, horizon) - the paying arm at each round; a subclass can make
        # it depend on the timestep
        self.stream_labels = self._stream_labels()
        # the correct arm always pays 1, so the means are one-hot over that label;
        # base.py reads this for regret, the optimal-arm rate and the plots
        self.arm_means = np.eye(self.num_arms)[self.stream_labels]

    def _stream_labels(self):
        """The paying arm at each (trial, timestep), shape (num_trials, horizon)."""
        return self.labels[self.stream]

    def get_context(self, trial, timestep):
        return self.features[self.stream[trial, timestep]]

    def get_contexts_batched(self, timestep):
        return self.features[self.stream[:, timestep]]

    def get_reward(self, trial, timestep, arm):
        return float(arm == self.stream_labels[trial, timestep])

    def get_rewards_batched(self, timestep, arms):
        return (arms == self.stream_labels[:, timestep]).astype(float)


class CovertypeCBEnv(ClassificationCBEnv):
    """Covertype served two ways: a binary vector for LinUCB, a line of text for an LLM."""

    def __init__(self, features, labels, arm_names, horizon, num_trials=20, seed=42):
        super().__init__(features, labels, horizon, num_trials=num_trials, seed=seed)

        if len(arm_names) != self.num_arms:
            raise ValueError(
                f"{len(arm_names)} arm names for {self.num_arms} classes"
            )
        self.arm_names = arm_names

    def render_context(self, context):
        """One context vector as a line of text, for a prompt."""
        # every block is one-hot, so the active bin is just its argmax
        parts = [
            f"{name}: {BIN_LABELS[np.argmax(context[i * NUM_BINS:(i + 1) * NUM_BINS])]}"
            for i, name in enumerate(FEATURE_NAMES)
        ]
        area = int(np.argmax(context[NUM_CONTINUOUS * NUM_BINS:]))
        parts.append(f"Wilderness area: {AREA_NAMES[area]}")

        return ", ".join(parts)

    def get_context_text(self, trial, timestep) -> str:
        """One trial's context at `timestep`, rendered for a prompt."""
        return self.render_context(self.get_context(trial, timestep))

    def get_context_texts_batched(self, timestep) -> list[str]:
        """Every trial's context at `timestep`, rendered for a prompt."""
        return [
            self.render_context(context)
            for context in self.get_contexts_batched(timestep)
        ]


class NonStationaryCovertypeCBEnv(CovertypeCBEnv):
    """Covertype with the label map rotated by one halfway through the horizon.

    The contexts and the arm names are untouched, so an agent that has learned the
    real cover types is wrong on every round after the switch until it relearns.
    """

    # same placeholders as the agent's default; the paying cover type is no longer
    # promised to be the true one, and the change point is withheld
    system_prompt = (
        "You are a contextual bandit algorithm that names the forest cover type of a site.\n"
        "Each round you are shown a description of one site, and you choose one of {0} cover types:\n"
        "{1}\n"
        "You get a reward of 1 if your choice is the one that pays for that site, and 0 if it is not.\n"
        "The cover type that pays for a given kind of site can change over time, so what paid "
        "earlier may stop paying, and a choice that failed earlier is worth trying again.\n"
        "You are told the reward for your own choice only; you are never told which choice pays.\n"
        "A good strategy to optimize for reward in these situations requires balancing exploration "
        "and exploitation. You need to explore to learn which descriptions go with which cover "
        "types, but you also have to exploit the information that you have to accumulate rewards. "
        "Recent rounds say more than old ones."
    )

    def _stream_labels(self):
        labels = super()._stream_labels()
        self.switch = self.horizon // 2

        return np.where(
            np.arange(self.horizon) < self.switch,
            labels,
            (labels + 1) % self.num_arms,
        )
