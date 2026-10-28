import asyncio

import pytest
from pytest import approx

from spar_harness.indivisible_good import indivisible_good_game
from spar_harness.referee import InvalidMove, run_episode
from spar_harness.schema import (
    Choice,
    EpisodeConfig,
    HorizonTreatment,
    IndivisibleOffer,
    Move,
    PartialType,
    PrivateType,
    TerminationSpec,
    Treatments,
    TypePrior,
    UniformPrior,
)

VERSIONS = ("strict", "lottery", "payment")


def fixed_priors():
    def prior(capability, cost):
        return TypePrior(
            capability=UniformPrior(low=capability, high=capability),
            cost_of_conflict=UniformPrior(low=cost, high=cost),
        )

    return {"alpha": prior(3.0, 0.1), "beta": prior(1.0, 0.2)}


def config(version="payment", max_rounds=20, stop_prob=0.0, status_quo=0.0, seed=0):
    return EpisodeConfig(
        episode_id="test",
        game=indivisible_good_game(
            version, max_rounds=max_rounds, stop_prob=stop_prob, status_quo=status_quo
        ),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors=fixed_priors(),
        seed=seed,
    )


def offer(version, taker="alpha"):
    """A valid proposal of the version that gives the taker 0.7 in value."""
    if version == "strict":
        return IndivisibleOffer(taker=taker)
    if version == "lottery":
        return IndivisibleOffer(alpha_chance=0.7 if taker == "alpha" else 0.3)
    return IndivisibleOffer(taker=taker, payment=0.3)


def scripted(*moves):
    """A player that plays the given moves in order."""
    remaining = list(moves)

    async def play(observation, context):
        return remaining.pop(0)

    return play


def stubborn(version):
    """Proposes the even offer and rejects every offer."""

    async def play(observation, context):
        if context.role == "proposer":
            if version == "lottery":
                return Move(action=IndivisibleOffer(alpha_chance=0.5))
            payment = 0.5 if version == "payment" else 0.0
            return Move(
                action=IndivisibleOffer(taker=observation.recipient, payment=payment)
            )
        return Move(action=Choice(kind="reject"))

    return play


def run(config, alpha, beta):
    return asyncio.run(run_episode(config, {"alpha": alpha, "beta": beta}))


@pytest.mark.parametrize("version", VERSIONS)
def test_accepted_proposal_ends_the_episode_in_agreement(version):
    state = run(
        config(version),
        alpha=scripted(Move(action=offer(version))),
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert state.ending == "agreement"
    assert len(state.rounds) == 1
    expected = (
        {"alpha": 1.0, "beta": 0.0}
        if version == "strict"
        else {"alpha": 0.7, "beta": 0.3}
    )
    assert state.payoffs == {
        player: approx(value) for player, value in expected.items()
    }


@pytest.mark.parametrize("version", VERSIONS)
def test_rejection_moves_play_to_the_next_round_with_the_other_proposer(version):
    state = run(
        config(version),
        alpha=scripted(Move(action=offer(version)), Move(action=Choice(kind="accept"))),
        beta=scripted(
            Move(action=Choice(kind="reject")), Move(action=offer(version, "beta"))
        ),
    )
    assert [r.proposer for r in state.rounds] == ["alpha", "beta"]
    assert state.ending == "agreement"
    expected = (
        {"alpha": 0.0, "beta": 1.0}
        if version == "strict"
        else {"alpha": 0.3, "beta": 0.7}
    )
    assert state.payoffs == {
        player: approx(value) for player, value in expected.items()
    }


@pytest.mark.parametrize("version", VERSIONS)
def test_either_player_can_fight(version):
    state = run(
        config(version),
        alpha=scripted(Move(action=Choice(kind="fight"))),
        beta=scripted(),
    )
    assert state.ending == "fight"
    assert state.payoffs == {"alpha": approx(0.65), "beta": approx(0.05)}
    state = run(
        config(version),
        alpha=scripted(Move(action=offer(version))),
        beta=scripted(Move(action=Choice(kind="fight"))),
    )
    assert state.ending == "fight"
    assert state.payoffs == {"alpha": approx(0.65), "beta": approx(0.05)}


@pytest.mark.parametrize("version", VERSIONS)
def test_horizon_and_random_stop_pay_the_status_quo_to_each_player(version):
    state = run(
        config(version, max_rounds=3), alpha=stubborn(version), beta=stubborn(version)
    )
    assert len(state.rounds) == 3
    assert state.ending == "horizon"
    assert state.payoffs == {"alpha": 0.0, "beta": 0.0}
    state = run(
        config(version, stop_prob=1.0, status_quo=0.5),
        alpha=stubborn(version),
        beta=stubborn(version),
    )
    assert len(state.rounds) == 1
    assert state.ending == "random_stop"
    assert state.payoffs == {"alpha": 0.5, "beta": 0.5}


def recording(player, log):
    """Wraps a player and records every observation it receives."""

    async def play(observation, context):
        log.append(observation)
        return await player(observation, context)

    return play


@pytest.mark.parametrize("version", VERSIONS)
def test_first_observation_briefs_the_player_on_the_game_and_its_version(version):
    log = []
    run(
        config(version, max_rounds=5, stop_prob=0.1, status_quo=0.0),
        alpha=recording(stubborn(version), log),
        beta=stubborn(version),
    )
    briefing = log[0].briefing
    assert briefing.version == version
    assert briefing.own_type == PrivateType(capability=3.0, cost_of_conflict=0.1)
    assert briefing.opponent_type == PartialType(capability=1.0, cost_of_conflict=0.2)
    assert briefing.opponent_prior is None
    assert briefing.status_quo == 0.0
    assert briefing.termination == TerminationSpec(max_rounds=5, stop_prob=0.1)
    assert log[1].briefing is None


def test_proposal_of_the_wrong_version_is_retried_with_the_reason():
    contexts = []

    async def alpha(observation, context):
        contexts.append(context)
        if len(contexts) == 1:
            return Move(action=IndivisibleOffer(alpha_chance=0.7))  # not strict
        return Move(action=IndivisibleOffer(taker="alpha"))

    state = run(
        config("strict"),
        alpha=alpha,
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert contexts[0].error is None
    assert "strict" in contexts[1].error
    assert state.rounds[0].moves["alpha"] == Move(
        action=IndivisibleOffer(taker="alpha")
    )
    assert state.rounds[0].defaulted == set()
    assert state.ending == "agreement"


def test_a_responder_cannot_propose_and_a_proposer_cannot_accept():
    contexts = []

    async def confused(observation, context):
        contexts.append(context)
        wrong = context.error is None
        if context.role == "proposer":
            if wrong:
                return Move(action=Choice(kind="accept"))
            return Move(action=IndivisibleOffer(taker=observation.recipient))
        if wrong:
            return Move(action=IndivisibleOffer(taker=observation.recipient))
        return Move(action=Choice(kind="reject"))

    state = run(config("strict", max_rounds=1), alpha=confused, beta=confused)
    errors = [context.error for context in contexts if context.error]
    assert "accept" in errors[0] and "proposal" in errors[0]
    assert "proposal" in errors[1] and "accept" in errors[1]
    assert state.ending == "horizon"


def broken(calls):
    """A player that can never produce a valid move."""

    async def play(observation, context):
        calls.append(context)
        raise InvalidMove("no tool call")

    return play


@pytest.mark.parametrize(
    "version, even",
    [
        ("strict", IndivisibleOffer(taker="alpha")),
        ("lottery", IndivisibleOffer(alpha_chance=0.5)),
        ("payment", IndivisibleOffer(taker="alpha", payment=0.5)),
    ],
)
def test_broken_proposer_makes_the_even_offer_after_two_retries(version, even):
    calls = []
    state = run(
        config(version),
        alpha=broken(calls),
        beta=scripted(Move(action=Choice(kind="accept"))),
    )
    assert len(calls) == 3
    assert state.rounds[0].moves["alpha"] == Move(action=even)
    assert state.rounds[0].defaulted == {"alpha"}
    assert state.ending == "agreement"


@pytest.mark.parametrize("version", VERSIONS)
def test_broken_responder_rejects_and_never_fights(version):
    state = run(config(version, max_rounds=1), alpha=stubborn(version), beta=broken([]))
    assert state.rounds[0].moves["beta"] == Move(action=Choice(kind="reject"))
    assert state.ending == "horizon"


@pytest.mark.parametrize("version", VERSIONS)
def test_proposer_that_breaks_later_repeats_its_last_proposal(version):
    first = Move(action=offer(version))

    async def alpha(observation, context):
        if context.round == 1:
            return first
        if context.role == "responder":
            return Move(action=Choice(kind="reject"))
        raise InvalidMove("no tool call")

    state = run(config(version, max_rounds=3), alpha=alpha, beta=stubborn(version))
    assert state.rounds[2].moves["alpha"] == first
    assert state.rounds[2].defaulted == {"alpha"}
