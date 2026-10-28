from pytest import approx

from spar_harness.indivisible_good import indivisible_good_game
from spar_harness.measures import indivisible_outcome_measures, player_measures
from spar_harness.schema import (
    Choice,
    EpisodeState,
    IndivisibleOffer,
    Move,
    PrivateType,
    RoundRecord,
)

TYPES = {
    "alpha": PrivateType(capability=3.0, cost_of_conflict=0.1),
    "beta": PrivateType(capability=1.0, cost_of_conflict=0.2),
}  # fight values 0.65 and 0.05


def choice(kind, **talk):
    return Move(action=Choice(kind=kind), **talk)


def agreed_on(offer, payoffs):
    return EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(
                round=1,
                proposer="alpha",
                moves={
                    "alpha": Move(action=IndivisibleOffer(taker="alpha")),
                    "beta": choice("reject"),
                },
            ),
            RoundRecord(
                round=2,
                proposer="beta",
                moves={"beta": Move(action=offer), "alpha": choice("accept")},
            ),
        ],
        ending="agreement",
        payoffs=payoffs,
    )


def test_agreement_is_in_range_when_both_deal_values_reach_the_fight_values():
    game = indivisible_good_game("payment", max_rounds=20, stop_prob=0.1)
    inside = agreed_on(
        IndivisibleOffer(taker="alpha", payment=0.1), {"alpha": 0.9, "beta": 0.1}
    )
    measures = indivisible_outcome_measures(game, inside)
    assert measures.agreement is True
    assert measures.fight is False
    assert measures.rounds_to_agreement == 2
    assert measures.efficiency == approx(1.0)
    assert measures.in_range is True
    outside = agreed_on(
        IndivisibleOffer(taker="alpha", payment=0.4), {"alpha": 0.6, "beta": 0.4}
    )
    assert indivisible_outcome_measures(game, outside).in_range is False


def test_strict_agreement_is_out_of_range_when_both_fight_values_are_positive():
    game = indivisible_good_game("strict", max_rounds=20, stop_prob=0.1)
    state = agreed_on(IndivisibleOffer(taker="beta"), {"alpha": 0.0, "beta": 1.0})
    assert indivisible_outcome_measures(game, state).in_range is False


def test_fight_loses_both_costs_of_conflict():
    game = indivisible_good_game("lottery", max_rounds=20, stop_prob=0.1)
    state = EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(round=1, proposer="alpha", moves={"alpha": choice("fight")})
        ],
        ending="fight",
        payoffs={"alpha": 0.65, "beta": 0.05},
    )
    measures = indivisible_outcome_measures(game, state)
    assert measures.fight is True
    assert measures.agreement is False
    assert measures.rounds_to_agreement is None
    assert measures.efficiency == approx(0.7)
    assert measures.in_range is None


def test_no_deal_has_zero_efficiency():
    game = indivisible_good_game("lottery", max_rounds=1, stop_prob=0.0)
    state = EpisodeState(
        types=TYPES,
        rounds=[
            RoundRecord(
                round=1,
                proposer="alpha",
                moves={
                    "alpha": Move(action=IndivisibleOffer(alpha_chance=0.5)),
                    "beta": choice("reject"),
                },
            )
        ],
        ending="horizon",
        payoffs={"alpha": 0.0, "beta": 0.0},
    )
    measures = indivisible_outcome_measures(game, state)
    assert measures.agreement is False
    assert measures.efficiency == 0.0
    assert measures.in_range is None


def test_declared_proposals_are_compared_by_deal_value():
    def gap_after(declared, actual):
        state = EpisodeState(
            types=TYPES,
            rounds=[
                RoundRecord(
                    round=1,
                    proposer="alpha",
                    moves={
                        "alpha": Move(action=IndivisibleOffer(taker="alpha")),
                        "beta": choice("reject"),
                    },
                ),
                RoundRecord(
                    round=2,
                    proposer="beta",
                    moves={
                        "beta": Move(action=IndivisibleOffer(taker="beta")),
                        "alpha": choice("reject", declared_action=declared),
                    },
                ),
                RoundRecord(
                    round=3,
                    proposer="alpha",
                    moves={"alpha": Move(action=actual), "beta": choice("accept")},
                ),
            ],
            ending="agreement",
            payoffs={"alpha": 0.5, "beta": 0.5},
        )
        return player_measures(state)["alpha"].intention_gap

    promised = IndivisibleOffer(taker="alpha", payment=0.3)
    assert gap_after(promised, IndivisibleOffer(taker="alpha", payment=0.305)) == 0.0
    assert gap_after(promised, IndivisibleOffer(taker="alpha", payment=0.1)) == 1.0
    assert gap_after(promised, IndivisibleOffer(taker="beta", payment=0.7)) == 0.0
