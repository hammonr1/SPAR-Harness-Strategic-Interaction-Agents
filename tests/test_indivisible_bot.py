import asyncio

import pytest
from pytest import approx

from spar_harness.bots import equilibrium_bot, indivisible_bot
from spar_harness.crisis_bargaining import (
    bargaining_range,
    crisis_bargaining_game,
    win_probability,
)
from spar_harness.indivisible_good import (
    deal_value,
    fight_values,
    indivisible_good_game,
)
from spar_harness.referee import run_episode
from spar_harness.schema import (
    EpisodeConfig,
    HorizonTreatment,
    IndivisibleOffer,
    Move,
    PartialType,
    PrivateInfoTreatment,
    PrivateType,
    TalkTreatment,
    Treatments,
    TypePrior,
    UniformPrior,
)

VERSIONS = ("strict", "lottery", "payment")


def fixed(value):
    return UniformPrior(low=value, high=value)


def priors(alpha, beta):
    """alpha and beta are (capability, cost_of_conflict) pairs."""
    return {
        player: TypePrior(capability=fixed(k), cost_of_conflict=fixed(c))
        for player, (k, c) in {"alpha": alpha, "beta": beta}.items()
    }


def types(alpha, beta):
    return {
        player: PrivateType(capability=k, cost_of_conflict=c)
        for player, (k, c) in {"alpha": alpha, "beta": beta}.items()
    }


def config(version, priors, status_quo=0.0, treatments=None):
    return EpisodeConfig(
        episode_id="test",
        game=indivisible_good_game(
            version, max_rounds=20, stop_prob=0.1, status_quo=status_quo
        ),
        treatments=treatments or Treatments(horizon=HorizonTreatment(known=True)),
        priors=priors,
        seed=0,
    )


def play(config, alpha=None, beta=None):
    players = {"alpha": alpha or indivisible_bot(), "beta": beta or indivisible_bot()}
    return asyncio.run(run_episode(config, players))


def first_offer(state):
    return state.rounds[0].moves["alpha"].action


COMPLETE_INFO_CASES = [
    ((3.0, 0.1), (1.0, 0.2)),
    ((1.0, 0.05), (1.0, 0.05)),
    ((1.0, 0.3), (9.0, 0.01)),
    ((5.0, 0.0), (2.0, 0.0)),  # costless conflict: the range is a single point
]


@pytest.mark.parametrize("alpha, beta", COMPLETE_INFO_CASES)
def test_payment_bots_agree_in_round_one_at_the_proposers_end_of_the_range(alpha, beta):
    state = play(config("payment", priors(alpha, beta), status_quo=0.5))
    assert state.ending == "agreement"
    assert len(state.rounds) == 1
    _, high = bargaining_range(win_probability(alpha[0], beta[0]), alpha[1], beta[1])
    assert deal_value(first_offer(state), "alpha") == approx(high)


@pytest.mark.parametrize("alpha, beta", COMPLETE_INFO_CASES)
def test_payment_bots_reproduce_crisis_bargaining_at_status_quo_one_half(alpha, beta):
    state = play(config("payment", priors(alpha, beta), status_quo=0.5))
    crisis = EpisodeConfig(
        episode_id="crisis",
        game=crisis_bargaining_game(max_rounds=20, stop_prob=0.1, status_quo=0.5),
        treatments=Treatments(horizon=HorizonTreatment(known=True)),
        priors=priors(alpha, beta),
        seed=0,
    )
    twin = asyncio.run(
        run_episode(crisis, {"alpha": equilibrium_bot(), "beta": equilibrium_bot()})
    )
    assert state.ending == twin.ending == "agreement"
    assert state.payoffs == {
        player: approx(value) for player, value in twin.payoffs.items()
    }


def test_payment_offer_names_the_taker_and_the_payment_from_alphas_side():
    # p = 0.75, beta's fight value is 0.05: alpha takes and pays 0.05
    state = play(config("payment", priors((3.0, 0.1), (1.0, 0.2))))
    offer = first_offer(state)
    assert offer.taker == "alpha"
    assert offer.payment == approx(0.05)
    # p = 0.1, beta's fight value is 0.89: beta takes and pays alpha 0.11
    state = play(config("payment", priors((1.0, 0.3), (9.0, 0.01))))
    offer = first_offer(state)
    assert offer.taker == "beta"
    assert offer.payment == approx(0.11)


@pytest.mark.parametrize("alpha, beta", COMPLETE_INFO_CASES)
def test_lottery_bots_agree_in_round_one_on_the_chance_at_betas_fight_value(
    alpha, beta
):
    state = play(config("lottery", priors(alpha, beta)))
    assert state.ending == "agreement"
    assert len(state.rounds) == 1
    _, high = bargaining_range(win_probability(alpha[0], beta[0]), alpha[1], beta[1])
    assert first_offer(state).alpha_chance == approx(high)


def test_strict_bot_claims_or_fights_but_never_concedes_when_fight_values_are_positive():
    positive = [
        (alpha, beta)
        for alpha, beta in COMPLETE_INFO_CASES
        if all(value > 0 for value in fight_values(types(alpha, beta)).values())
    ]
    assert len(positive) == 3
    for alpha, beta in positive:
        state = play(config("strict", priors(alpha, beta)))
        assert first_offer(state) == IndivisibleOffer(taker="alpha"), (alpha, beta)
        assert state.ending == "fight"  # beta fights rather than take nothing
        assert len(state.rounds) == 1
    # as responder, facing a concession, it accepts; facing a claim, it fights
    claims, concedes = IndivisibleOffer(taker="alpha"), IndivisibleOffer(taker="beta")
    for offer, expected in ((claims, "fight"), (concedes, "accept")):

        async def alpha(observation, context, offer=offer):
            return Move(action=offer)

        state = play(config("strict", priors((3.0, 0.1), (1.0, 0.2))), alpha=alpha)
        assert state.rounds[0].moves["beta"].action.kind == expected


def test_strict_bots_agree_when_the_other_side_would_lose_from_fighting():
    # p = 0.9 and c_beta = 0.2: beta's fight value is -0.1, so it takes nothing
    state = play(config("strict", priors((9.0, 0.1), (1.0, 0.2))))
    assert first_offer(state) == IndivisibleOffer(taker="alpha")
    assert state.ending == "agreement"
    assert state.payoffs == {"alpha": 1.0, "beta": 0.0}


def test_strict_bot_concedes_when_claiming_cannot_beat_conceding():
    # p = 0.1 and c_alpha = 0.3: alpha's fight value is -0.2. Claiming leads to a
    # fight worth -0.2; conceding is worth 0 and is the safer offer.
    state = play(config("strict", priors((1.0, 0.3), (9.0, 0.01))))
    assert first_offer(state) == IndivisibleOffer(taker="beta")
    assert state.ending == "agreement"


def private_cost_config(version):
    """The worked example of the spec: p = 0.5, c_alpha = 0.1 and a private
    c_beta ~ U[0, 0.4]. The optimal concession is 0.35, so alpha keeps 0.65."""
    priors = {
        "alpha": TypePrior(capability=fixed(1.0), cost_of_conflict=fixed(0.1)),
        "beta": TypePrior(
            capability=fixed(1.0), cost_of_conflict=UniformPrior(low=0.0, high=0.4)
        ),
    }
    treatments = Treatments(
        horizon=HorizonTreatment(known=True),
        private_info=PrivateInfoTreatment(private_fields={"cost_of_conflict"}),
    )
    return config(version, priors, treatments=treatments)


def test_under_private_information_the_bot_trades_risk_for_value():
    state = play(private_cost_config("lottery"))
    assert first_offer(state).alpha_chance == approx(0.65, abs=0.002)
    state = play(private_cost_config("payment"))
    offer = first_offer(state)
    assert offer.taker == "alpha"
    assert offer.payment == approx(0.35, abs=0.002)


def test_bot_declares_its_private_fields_truthfully_when_it_may():
    private = private_cost_config("lottery")
    talk = TalkTreatment(messages=False, declared_action=False, declared_type=True)
    config = private.model_copy(
        update={"treatments": private.treatments.model_copy(update={"talk": talk})}
    )
    state = play(config)
    beta_cost = state.types["beta"].cost_of_conflict
    move = state.rounds[0].moves["beta"]
    assert move.declared_type == PartialType(cost_of_conflict=beta_cost)
    assert move.message is None
