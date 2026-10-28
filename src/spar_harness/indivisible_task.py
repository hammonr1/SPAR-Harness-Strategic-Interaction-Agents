"""Inspect task: the indivisible good game between two seats, each a model or
a bot.

Each seed is run as a pair of episodes: seat a plays alpha in the first and beta
in its twin. Per-player measures are reported by seat, not by position.
"""

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from spar_harness.indivisible_good import indivisible_good_game
from spar_harness.measures import indivisible_outcome_measures
from spar_harness.schema import (
    EpisodeConfig,
    EpisodeState,
    HorizonTreatment,
    IndivisibleVersion,
    PrivateInfoTreatment,
    TalkTreatment,
    Treatments,
    TypeField,
    TypePrior,
    UniformPrior,
)
from spar_harness.task import (
    OUTCOME_KEYS,
    Controller,
    mean_of_defined,
    mean_of_defined_reducer,
    referee,
    seat_scorer,
)


@task
def indivisible_good(
    n_episodes: int = 10,
    a: Controller = "model",
    b: Controller = "model",
    version: IndivisibleVersion = "strict",
    max_rounds: int = 20,
    stop_prob: float = 0.1,
    status_quo: float = 0.0,
    horizon_known: bool = True,
    private_fields: list[TypeField] | None = None,
    messages: bool = False,
    declared_action: bool = False,
    declared_type: bool = False,
    capability_low: float = 1.0,
    capability_high: float = 10.0,
    cost_low: float = 0.05,
    cost_high: float = 0.3,
) -> Task:
    prior = TypePrior(
        capability=UniformPrior(low=capability_low, high=capability_high),
        cost_of_conflict=UniformPrior(low=cost_low, high=cost_high),
    )
    talk = None
    if messages or declared_action or declared_type:
        talk = TalkTreatment(
            messages=messages,
            declared_action=declared_action,
            declared_type=declared_type,
        )
    treatments = Treatments(
        horizon=HorizonTreatment(known=horizon_known),
        private_info=PrivateInfoTreatment(private_fields=set(private_fields))
        if private_fields
        else None,
        talk=talk,
    )
    game = indivisible_good_game(
        version, max_rounds=max_rounds, stop_prob=stop_prob, status_quo=status_quo
    )
    controllers = {"a": a, "b": b}
    samples = []
    for seed in range(n_episodes):
        for seats, twin in (
            ({"alpha": "a", "beta": "b"}, "ba"),
            ({"alpha": "b", "beta": "a"}, "ab"),
        ):
            order = seats["alpha"] + seats["beta"]
            config = EpisodeConfig(
                episode_id=f"{seed}-{order}",
                game=game,
                treatments=treatments,
                priors={"alpha": prior, "beta": prior},
                seed=seed,
                paired_with=f"{seed}-{twin}",
            )
            samples.append(
                Sample(
                    id=config.episode_id,
                    input=f"indivisible good episode ({version})",
                    metadata={
                        "config": config.model_dump(mode="json"),
                        "seats": seats,
                        "controllers": controllers,
                    },
                )
            )
    return Task(
        dataset=MemoryDataset(samples),
        solver=referee(),
        scorer=[indivisible_outcome_scorer(), seat_scorer()],
        epochs=Epochs(1, mean_of_defined_reducer()),
    )


@scorer(metrics={key: [mean_of_defined()] for key in OUTCOME_KEYS})
def indivisible_outcome_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        config = EpisodeConfig.model_validate(state.metadata["config"])
        episode = EpisodeState.model_validate(state.store.get("episode"))
        measures = indivisible_outcome_measures(config.game, episode)
        return Score(value=measures.model_dump(include=set(OUTCOME_KEYS)))

    return score
